"""The wire contract: what a caller may send, and what comes back.

Three principles, in the order they resolve conflicts:

1. **The wire vocabulary is the dataset vocabulary.** Field names are the raw CSV
   column names verbatim -- `MonthlyCharges`, not `monthly_charges` -- and so are the
   category levels. No alias layer, no rename map, and nothing to keep in sync when a
   column changes. It costs a little PEP 8 in the field names and buys the property that
   a row of `data/Telco-Customer-Churn.csv` is a valid request body -- which is why the
   two columns `data.clean` normalises are normalised here too, by the same rules.
2. **Strict about structure, honest about the model.** `extra="forbid"` and closed
   `Literal` levels mean a typo is a 422 that names the allowed values, not a silently
   degraded score. See the level-validation note below -- that one is a live decision.
3. **The dataclasses in `scoring` and `explain` are already the right shape.** These
   models are a serialisation layer over `Score` and `Driver`, not a second model of the
   domain. `from_driver` exists only because SHAP hands back numpy scalars; everything
   else maps field for field.

**Closed levels -- open decision, provisional default.** `features.py` sets
`handle_unknown="ignore"` precisely so an unseen level scores rather than raising, and
calls that "a slightly-off score beats a 500". Closed `Literal`s at the edge partly undo
that: a genuinely new product tier would now 422. The default here is closed anyway,
because the failure this actually catches is `"month-to-month"` or `"Two Year"` from a
caller's casing bug -- which `handle_unknown="ignore"` turns into an all-zeros encoding
and a confidently wrong score, the worst of the three outcomes. The tolerance in
`features.py` still covers the offline and batch paths, which have no Pydantic layer.
Reopening this means widening these fields to `str`, not touching `features.py`.

**`MAX_BATCH_SIZE` is measured, not guessed** -- see the constant.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from churn.actions import ActionId
from churn.explain import Driver
from churn.scoring import RiskBand, Score

# Gate: batch cap, resolved (S4). Compute is not what bounds this. Measured against the
# real 276-tree artifact, scoring costs ~16ms fixed plus ~0.02ms per record and Pydantic
# ~8us per record, so a 1000-record request is ~40ms warm against a 30s timeout -- record
# 101 costs 20 microseconds and a cap of 100 would buy nothing.
#
# The binding limit is Lambda's 6 MB synchronous request payload, which at ~488 bytes per
# record is reached around 12,900 records. 1000 sits an order of magnitude below it, so a
# caller who overshoots gets this schema's 422 naming the limit rather than an opaque
# rejection from the platform before any of our code runs.
MAX_BATCH_SIZE = 1000

YesNo = Literal["No", "Yes"]

# `No internet service` is a distinct level in the data, not a second spelling of `No`.
# The encoder was fitted with them separate and the model uses the difference, so the
# contract keeps them separate too.
ServiceOption = Literal["No", "Yes", "No internet service"]


def _normalise_senior_citizen(value: Any) -> Any:
    """Accept the raw CSV's 0/1 for `SeniorCitizen` and hand on the cleaned Yes/No.

    This is the one column the dataset itself is inconsistent about -- `data.clean`
    normalises it for training, and a caller reading the CSV would otherwise send `1`
    and get a 422 on a value the source file told them was correct. Booleans are
    accepted for the same reason: JSON clients reach for `true` here.

    `"0"` and `"1"` are accepted alongside `0` and `1` because a CSV has no types. Read
    the file with pandas and the column arrives as `int64`; read it with `csv.DictReader`
    and every field is a string. Both are the same row, so both are valid. Pydantic
    already coerces the quoted numerics (`"132.4"` for `TotalCharges`) for free -- this
    is the same leniency, applied to the one field whose target type is a `Literal` and
    therefore gets no coercion of its own.
    """
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int) and value in (0, 1):
        return "Yes" if value else "No"
    if value in ("0", "1"):
        return "Yes" if value == "1" else "No"
    return value


SeniorCitizenField = Annotated[YesNo, BeforeValidator(_normalise_senior_citizen)]


def _normalise_total_charges(value: Any) -> Any:
    """Accept the raw CSV's blank `TotalCharges` and hand on the cleaned 0.0.

    The other column the dataset is inconsistent about, and the same reasoning as
    `_normalise_senior_citizen`: `data.clean` already resolved blank to 0.0 in S1, on the
    grounds that every blank row is a tenure-0 customer who has genuinely been billed
    nothing. This applies that resolved decision at the edge rather than making a new
    one, so the training path and the wire path agree on what a blank means. Deliberately
    unconditional, exactly like `clean` -- no `tenure == 0` guard it does not have.
    """
    if isinstance(value, str) and not value.strip():
        return 0.0
    return value


TotalChargesField = Annotated[float, BeforeValidator(_normalise_total_charges), Field(ge=0)]


class CustomerRecord(BaseModel):
    """One customer, in the shape the fitted preprocessor expects.

    `customerID` is optional, echoed back, and never a model input -- `data.clean` drops
    it and the transformer's `remainder="drop"` would ignore it regardless. It is here
    because `/score` returns a list and input order is otherwise the only way a caller
    can join results back to their own records. It must not reach the logs.
    """

    model_config = ConfigDict(extra="forbid")

    customerID: str | None = None

    gender: Literal["Female", "Male"]
    SeniorCitizen: SeniorCitizenField
    Partner: YesNo
    Dependents: YesNo

    # No upper bound on tenure: the training data tops out at 72 months, but a
    # longer-tenured customer is out of distribution, not invalid, and refusing to score
    # them would be the wrong answer to a real record.
    tenure: int = Field(ge=0)

    PhoneService: YesNo
    MultipleLines: Literal["No", "Yes", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: ServiceOption
    OnlineBackup: ServiceOption
    DeviceProtection: ServiceOption
    TechSupport: ServiceOption
    StreamingTV: ServiceOption
    StreamingMovies: ServiceOption

    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]

    MonthlyCharges: float = Field(ge=0)
    TotalCharges: TotalChargesField

    def to_record(self) -> dict[str, Any]:
        """The mapping `score` and `explain` take. Drops the identifier."""
        return self.model_dump(exclude={"customerID"})


class ScoreRequest(BaseModel):
    """List in, list out -- always, even for one customer.

    A union of "record or list of records" would make the *response* shape depend on the
    request shape, which is a worse contract than one extra level of nesting.
    """

    model_config = ConfigDict(extra="forbid")

    customers: list[CustomerRecord] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class SingleCustomerRequest(BaseModel):
    """Body for the one-record endpoints. `explain` rejects a batch by design.

    Wrapped rather than taking a bare `CustomerRecord` so `/explain` and `/recommend`
    read like `/score`, and so a future `top_n` has somewhere to live.
    """

    model_config = ConfigDict(extra="forbid")

    customer: CustomerRecord


class ScoredCustomer(BaseModel):
    """One row of a `/score` response, in request order."""

    customerID: str | None = None
    probability: float
    band: RiskBand

    @classmethod
    def from_score(cls, score: Score, customer_id: str | None = None) -> "ScoredCustomer":
        return cls(customerID=customer_id, probability=score.probability, band=score.band)


class DriverModel(BaseModel):
    """One SHAP driver.

    `contribution` is in pre-calibration margin units while the `probability` alongside
    it is calibrated -- the `explain` module docstring has the why. `direction` is about
    churn risk, so a protective driver reads `decreases` even though its contribution is
    negative.
    """

    feature: str
    value: str | int | float | None
    contribution: float
    direction: Literal["increases", "decreases"]

    @classmethod
    def from_driver(cls, driver: Driver) -> "DriverModel":
        """Not `model_validate(asdict(driver))`: `Driver.value` is read off a pandas row,
        so it arrives as a numpy scalar, which Pydantic will not coerce to `int`.
        `.item()` is the documented way back to a Python builtin; anything else passes
        through untouched.
        """
        value = driver.value
        if hasattr(value, "item"):
            value = value.item()
        return cls(
            feature=driver.feature,
            value=value,
            contribution=driver.contribution,
            direction=driver.direction,
        )


class ScoreResponse(BaseModel):
    model_version: str
    results: list[ScoredCustomer]


class ExplainResponse(BaseModel):
    model_version: str
    customerID: str | None = None
    probability: float
    band: RiskBand
    drivers: list[DriverModel]


class RecommendResponse(BaseModel):
    """`/explain` plus a retention action, or plus a reason there is none.

    `action` and `reason` are mutually exclusive and one of them is always populated. The
    score and drivers are present either way -- that is the graceful-degradation contract
    from `SPEC.md` §5: an LLM outage costs the caller the recommendation, never the
    scoring.

    `drivers_used` is the subset of `drivers` the model cited. It is validated against
    what was actually sent, so a recommendation cannot cite evidence it was never given.
    """

    model_version: str
    customerID: str | None = None
    probability: float
    band: RiskBand
    drivers: list[DriverModel]

    action: ActionId | None = None
    rationale: str | None = None
    drivers_used: list[str] = Field(default_factory=list)
    # Populated only when `action` is null. A stable identifier, not prose, so a caller
    # can branch on it -- see `recommend.DegradationReason`.
    reason: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_version: str

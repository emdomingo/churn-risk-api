"""One customer record in, its top SHAP drivers out.

**The contributions are in margin units, not probability units.** `TreeExplainer` runs on
the booster inside the calibrator (see `ChurnArtifact.booster`), so a contribution of
+0.84 means "this feature pushed the log-odds up by 0.84 *before* calibration". The score
returned alongside it is calibrated. Calibration is a monotone map, so the *ranking* and
the *signs* are unaffected — which is what makes this defensible — but the magnitudes do
not add up to the probability, and the README must say so rather than imply otherwise.

**Drivers are named at the level, not the column** (gate #3, resolved): the encoder emits
`Contract_Month-to-month`, and this module reports it as `Contract=Month-to-month`.
Aggregating the levels back onto `Contract` would give shorter lists, but `/recommend`
needs the level to choose an action — "Contract" alone cannot tell the model whether a
contract-term incentive is the right offer or a pointless one. The name is reconstructed
from the fitted `OneHotEncoder.categories_` rather than by splitting the encoded name on
its first underscore, so it stays correct if a source column is ever renamed.

Note that `drop=None` in the encoder means a customer carries a contribution for levels
they do *not* have. A month-to-month customer can show `Contract=Two year` as a strongly
protective driver — the model is saying the absence of that contract is what costs them.
That is real signal, and it is why selecting drivers is a decision and not just a sort.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer

from churn.artifact import ChurnArtifact
from churn.scoring import as_frame

TOP_N_DRIVERS = 3

# A contribution smaller than this fraction of the customer's own largest one is treated
# as noise and never reported. See `select_drivers` for where the number comes from.
RELATIVE_NOISE_FLOOR = 0.05

# `Contract=Month-to-month` reads as an assertion about the customer; the encoder's own
# `Contract_Month-to-month` reads as a column name. The API returns the former.
LEVEL_SEPARATOR = "="


@dataclass(frozen=True)
class Driver:
    """One line of the `/explain` response."""

    feature: str  # level-specific: "Contract=Month-to-month", or plain "tenure"
    value: Any  # the customer's raw value on the source column
    contribution: float  # SHAP value, in pre-calibration margin units
    direction: str  # "increases" or "decreases" — churn risk, not the score's sign


def source_columns(preprocessor: ColumnTransformer) -> dict[str, tuple[str, str | None]]:
    """Map each encoded column back to `(source column, level)`.

    Read off the fitted transformers rather than parsed out of the encoded name: the
    one-hot encoder already knows which levels it produced for which column, and asking
    it is the only version that cannot be broken by a column name containing the
    separator it was joined with.
    """
    mapping: dict[str, tuple[str, str | None]] = {}
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue
        # A fitted encoder exposes `categories_`; the passthrough branch does not. Testing
        # for the attribute rather than for `== "passthrough"` is deliberate: `set_output`
        # replaces that string with a `FunctionTransformer` at fit time.
        levels_by_column = getattr(transformer, "categories_", None)
        if levels_by_column is None:
            mapping.update({column: (column, None) for column in columns})
            continue
        for column, levels in zip(columns, levels_by_column, strict=True):
            mapping.update({f"{column}_{level}": (column, str(level)) for level in levels})
    return mapping


def driver_name(encoded_column: str, mapping: dict[str, tuple[str, str | None]]) -> str:
    """`Contract_Two year` -> `Contract=Two year`; `tenure` -> `tenure`."""
    column, level = mapping.get(encoded_column, (encoded_column, None))
    return column if level is None else f"{column}{LEVEL_SEPARATOR}{level}"


def contributions(artifact: ChurnArtifact, record: Mapping | pd.DataFrame) -> pd.Series:
    """Every feature's SHAP value for one record, indexed by encoded column name.

    All 46 of them, unranked and unfiltered — deciding which ones a caller should see is
    `select_drivers`'s job, kept separate so that policy is testable on a hand-built
    Series with no model involved.
    """
    encoded = artifact.transform(as_frame(record))
    if len(encoded) != 1:
        raise ValueError(f"explain expects exactly one record, got {len(encoded)}")
    values = artifact.explainer.shap_values(encoded)
    return pd.Series(values[0], index=encoded.columns, dtype="float64")


def select_drivers(values: pd.Series, top_n: int = TOP_N_DRIVERS) -> pd.Series:
    """The `top_n` largest contributions by magnitude, noise floor applied, signs kept.

    Ranked by `abs()` rather than filtered to positive contributions: for a low-risk
    customer the honest answer to "what drove this score" is the factors keeping them,
    and returning three weak positive contributions instead would manufacture a risk
    story the model never told. `/recommend` receives the signs along with the values, so
    it can tell it is looking at a customer who is fine.

    The floor is **relative to this customer's own strongest driver**, not an absolute
    cutoff. Across the test split the median contribution is ~0.0005 while every
    customer's largest one is at least 0.365 — most of the 46 features per record are
    genuinely nothing (a one-hot level the customer does not have contributes exactly
    0.0), but "nothing" sits at a different scale for a customer with one dominant driver
    than for one with six comparable ones. A relative floor self-scales; a fixed number
    would have to be picked by feel.

    It is a guard, not a filter. At 5%, no customer in the 1,409-row test split loses a
    driver to it, so on today's data it never fires. It exists so that a genuinely
    featureless customer returns two drivers rather than padding the list with a
    contribution of +0.004 that `/recommend` would then be asked to justify.
    """
    magnitudes = values.abs()
    strongest = magnitudes.max()
    if not strongest > 0:  # every contribution exactly zero — nothing worth reporting
        return values.iloc[:0]

    # `kind="stable"` so ties break on feature-matrix order rather than on whatever the
    # default sort happens to do: two customers with identical contributions must get
    # identical responses.
    ranked = magnitudes.sort_values(ascending=False, kind="stable")
    surviving = ranked[ranked >= RELATIVE_NOISE_FLOOR * strongest].head(top_n)
    return values.reindex(surviving.index)


def explain(
    artifact: ChurnArtifact, record: Mapping | pd.DataFrame, top_n: int = TOP_N_DRIVERS
) -> list[Driver]:
    """Top drivers for one record, ready to serialise or hand to `/recommend`."""
    frame = as_frame(record)
    mapping = source_columns(artifact.preprocessor)
    selected = select_drivers(contributions(artifact, frame), top_n=top_n)

    drivers = []
    for encoded_column, contribution in selected.items():
        source_column, _ = mapping.get(str(encoded_column), (str(encoded_column), None))
        drivers.append(
            Driver(
                feature=driver_name(str(encoded_column), mapping),
                value=frame.iloc[0].get(source_column),
                contribution=float(contribution),
                direction="increases" if contribution > 0 else "decreases",
            )
        )
    return drivers

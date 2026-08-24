"""Contract tests for the HTTP edge.

These assert the *contract*, not the model: that valid input is accepted, invalid input
is refused with a useful status, responses carry the fields the README promises, and
nothing about a customer reaches the logs. Model quality is `test_train.py`'s job and
band boundaries are `test_scoring.py`'s -- duplicating them here would mean two places to
update when a number changes.
"""

import logging

import pytest
from fastapi import HTTPException

from churn.api import artifact_dependency
from churn.logging_setup import LOGGABLE_FIELDS, JsonFormatter
from churn.schemas import MAX_BATCH_SIZE


@pytest.fixture
def customer(raw_rows: list[dict]) -> dict:
    """One valid request body, taken straight from the CSV fixture."""
    return dict(raw_rows[0])


# --- health ----------------------------------------------------------------------


def test_health_reports_the_loaded_model_version(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "test-artifact"}


def test_missing_artifact_is_503_not_500(monkeypatch):
    """A model that was never built is "not ready", not "crashed".

    Exercised on the dependency directly: the whole point is what happens when no
    artifact exists, which the `client` fixture deliberately papers over.
    """
    import churn.api

    def raise_missing():
        raise FileNotFoundError("no artifact")

    monkeypatch.setattr(churn.api, "get_artifact", raise_missing)

    with pytest.raises(HTTPException) as excinfo:
        artifact_dependency()

    assert excinfo.value.status_code == 503


# --- /score ----------------------------------------------------------------------


def test_score_returns_one_result_per_customer_in_request_order(client, raw_rows):
    batch = [dict(row) for row in raw_rows[:5]]

    response = client.post("/score", json={"customers": batch})

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "test-artifact"
    assert [result["customerID"] for result in body["results"]] == [
        row["customerID"] for row in batch
    ]


def test_score_probabilities_are_in_range_with_a_matching_band(client, customer):
    response = client.post("/score", json={"customers": [customer]})

    result = response.json()["results"][0]
    assert 0.0 <= result["probability"] <= 1.0
    assert result["band"] in {"low", "medium", "high"}


def test_score_accepts_a_single_customer_as_a_one_item_list(client, customer):
    """There is no "one record" shape. List in, list out, always."""
    response = client.post("/score", json={"customers": [customer]})

    assert response.status_code == 200
    assert len(response.json()["results"]) == 1


def test_score_rejects_an_empty_batch(client):
    assert client.post("/score", json={"customers": []}).status_code == 422


def test_score_rejects_a_batch_over_the_cap(client, customer):
    """The cap is enforced at validation, before any scoring work happens."""
    oversized = [dict(customer) for _ in range(MAX_BATCH_SIZE + 1)]

    response = client.post("/score", json={"customers": oversized})

    assert response.status_code == 422
    assert "at most" in response.text


def test_score_accepts_a_full_batch_at_the_cap(client, customer):
    response = client.post(
        "/score", json={"customers": [dict(customer) for _ in range(MAX_BATCH_SIZE)]}
    )

    assert response.status_code == 200
    assert len(response.json()["results"]) == MAX_BATCH_SIZE


# --- validation ------------------------------------------------------------------


def test_every_raw_fixture_row_is_a_valid_request_body(client, raw_rows):
    """The claim `schemas` makes: a row of the CSV can be posted unmodified.

    This is what pins the two normalisations -- 0/1 `SeniorCitizen` and blank
    `TotalCharges` -- to the same rules `data.clean` applies at training time.
    """
    response = client.post("/score", json={"customers": [dict(row) for row in raw_rows]})

    assert response.status_code == 200
    assert len(response.json()["results"]) == len(raw_rows)


def test_blank_total_charges_is_read_as_zero(client, customer):
    customer["TotalCharges"] = " "
    customer["tenure"] = 0

    assert client.post("/score", json={"customers": [customer]}).status_code == 200


@pytest.mark.parametrize(
    "raw,expected",
    [(0, "No"), (1, "Yes"), (True, "Yes"), ("0", "No"), ("1", "Yes"), ("Yes", "Yes")],
)
def test_senior_citizen_accepts_the_raw_csv_encoding(customer, raw, expected):
    """The quoted forms matter as much as the bare ones.

    `raw_rows` comes from pandas, which types this column `int64`, so a JSON body built
    from it carries a bare `0`. A caller using `csv.DictReader` sends `"0"` instead --
    same row, no types in the file. Only the bare form was covered until a live request
    failed on the quoted one.
    """
    from churn.schemas import CustomerRecord

    customer["SeniorCitizen"] = raw

    assert CustomerRecord.model_validate(customer).SeniorCitizen == expected


def test_a_csv_module_row_is_a_valid_request_body(client):
    """The other half of `test_every_raw_fixture_row_is_a_valid_request_body`: same file,
    read with the stdlib instead of pandas, so every field arrives as a string."""
    import csv
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "telco_sample.csv"
    with open(fixture, newline="") as handle:
        rows = [row for row in csv.DictReader(handle)]
    for row in rows:
        row.pop("Churn")

    response = client.post("/score", json={"customers": rows})

    assert response.status_code == 200, response.text
    assert len(response.json()["results"]) == len(rows)


def test_a_mis_cased_level_is_refused_rather_than_silently_degraded(client, customer):
    """`handle_unknown="ignore"` would encode this as all-zeros and score it anyway.

    That is the failure the closed `Literal`s exist to catch: a confidently wrong score
    is worse than a 422 that names the allowed values.
    """
    customer["Contract"] = "month-to-month"

    response = client.post("/score", json={"customers": [customer]})

    assert response.status_code == 422
    assert "Contract" in response.text


def test_an_unknown_field_is_refused(client, customer):
    customer["MonthlyChargez"] = 42.0

    response = client.post("/score", json={"customers": [customer]})

    assert response.status_code == 422
    assert "MonthlyChargez" in response.text


def test_a_missing_field_is_refused(client, customer):
    del customer["Contract"]

    assert client.post("/score", json={"customers": [customer]}).status_code == 422


def test_negative_charges_are_refused(client, customer):
    customer["MonthlyCharges"] = -1.0

    assert client.post("/score", json={"customers": [customer]}).status_code == 422


# --- /explain --------------------------------------------------------------------


def test_explain_returns_score_and_drivers_for_one_customer(client, customer):
    response = client.post("/explain", json={"customer": customer})

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "test-artifact"
    assert body["customerID"] == customer["customerID"]
    assert 0.0 <= body["probability"] <= 1.0
    assert 1 <= len(body["drivers"]) <= 3


def test_driver_direction_matches_the_sign_of_its_contribution(client, customer):
    drivers = client.post("/explain", json={"customer": customer}).json()["drivers"]

    for driver in drivers:
        expected = "increases" if driver["contribution"] > 0 else "decreases"
        assert driver["direction"] == expected


def test_drivers_are_ranked_by_magnitude(client, customer):
    """The selection policy is `explain`'s; this only checks it survives serialisation."""
    drivers = client.post("/explain", json={"customer": customer}).json()["drivers"]

    magnitudes = [abs(driver["contribution"]) for driver in drivers]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_explain_refuses_a_batch_shaped_body(client, customer):
    """`explain` rejects a batch rather than explaining row 0; the schema says so first."""
    response = client.post("/explain", json={"customers": [customer]})

    assert response.status_code == 422


# --- request correlation and logging ---------------------------------------------


def test_the_callers_request_id_is_echoed_back(client, customer):
    response = client.post(
        "/score", json={"customers": [customer]}, headers={"x-request-id": "abc-123"}
    )

    assert response.headers["x-request-id"] == "abc-123"


def test_a_request_id_is_generated_when_the_caller_sends_none(client):
    assert client.get("/health").headers["x-request-id"]


def test_the_formatter_drops_fields_outside_the_allowlist():
    """Redaction is structural: a PII field attached to a log call never reaches stdout.

    This is the test that makes the allowlist a guarantee rather than a convention.
    """
    record = logging.LogRecord("churn.api", logging.INFO, "", 0, "scored", None, None)
    record.customerID = "0305-SQECB"
    record.MonthlyCharges = 54.6
    record.request_id = "abc-123"

    line = JsonFormatter().format(record)

    assert "0305-SQECB" not in line
    assert "MonthlyCharges" not in line
    assert "abc-123" in line


def test_no_customer_field_is_loggable():
    """The allowlist and the request schema must not overlap. If a future field is added
    to both, this fails before it can ship."""
    from churn.schemas import CustomerRecord

    assert LOGGABLE_FIELDS.isdisjoint(CustomerRecord.model_fields)


# --- /recommend ------------------------------------------------------------------


@pytest.fixture
def llm_client(client):
    """Swap the Anthropic client for a fake, per test.

    Same seam as the artifact -- `app.dependency_overrides` -- which is why S5 needed no
    new test scaffolding. Returns a setter so each test picks its own response.
    """
    from churn.api import app, get_llm_client

    def use(fake):
        app.dependency_overrides[get_llm_client] = lambda: fake
        return client

    yield use
    app.dependency_overrides.pop(get_llm_client, None)


def test_recommend_returns_score_drivers_and_an_action(llm_client, customer):
    from fakes import draft, responding

    fake = responding(draft(action="contract_term_incentive"))
    response = llm_client(fake).post("/recommend", json={"customer": customer})

    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "contract_term_incentive"
    assert body["rationale"]
    assert body["reason"] is None
    assert body["drivers"]
    assert body["probability"] == pytest.approx(body["probability"])


def test_recommend_degrades_to_200_when_the_llm_fails(llm_client, customer):
    """The contract that matters: an LLM outage costs the recommendation, never the
    scoring, and never returns a 5xx."""
    import anthropic
    import httpx

    from fakes import failing

    failure = anthropic.APITimeoutError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    response = llm_client(failing(failure)).post("/recommend", json={"customer": customer})

    assert response.status_code == 200
    body = response.json()
    assert body["action"] is None
    assert body["reason"] == "llm_timeout"
    # The parts that need no network are still there.
    assert 0.0 <= body["probability"] <= 1.0
    assert body["drivers"]


def test_recommend_degrades_to_200_when_no_api_key_is_configured(llm_client, customer):
    response = llm_client(None).post("/recommend", json={"customer": customer})

    assert response.status_code == 200
    assert response.json()["reason"] == "llm_not_configured"


def test_an_off_catalogue_action_never_reaches_the_response(llm_client, customer):
    from fakes import draft, responding

    fake = responding(draft(action="six_months_free"))
    response = llm_client(fake).post("/recommend", json={"customer": customer})

    assert response.status_code == 200
    body = response.json()
    assert body["action"] is None
    assert body["reason"] == "action_not_in_catalogue"


def test_recommend_cites_only_drivers_from_the_same_response(llm_client, customer):
    """`drivers_used` is auditable: every entry appears in `drivers`."""
    from fakes import draft, responding

    fake = responding(draft(action="fee_waiver"))
    body = llm_client(fake).post("/recommend", json={"customer": customer}).json()

    features = {driver["feature"] for driver in body["drivers"]}
    assert set(body["drivers_used"]) <= features

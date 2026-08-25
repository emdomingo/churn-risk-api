"""The one test that talks to the network.

Every other test in this suite is hermetic -- `conftest.py` trains a tiny model on the
committed fixture and nothing reaches AWS. This file is the deliberate exception: it
exercises the *deployed* service, which is the only way to catch the failures that live
between a green build and a working URL (a bad image manifest, a missing env var, an API
Gateway route that was never wired).

**It is opt-in and skips by default.** `CHURN_SMOKE_URL` must be set, so `uv run pytest`
stays offline and `ci.yml` never depends on a live endpoint:

    $env:CHURN_SMOKE_URL = "https://03bmm97sm6.execute-api.ap-southeast-2.amazonaws.com"
    uv run pytest tests/test_smoke.py -v

Set `CHURN_SMOKE_VERSION` as well to assert a specific `model_version` -- that is the
deploy-verification form, and it is what `deploy.yml` does inline after a stack update.

**The first request after a deploy returns 503 and that is expected**, so every request
here retries. See gate 10 in `PLAN.md`: init overruns Lambda's 10s cap, is suppressed and
re-run inside the invocation, and the total overruns API Gateway's 29s. Retrying is not
papering over a flake -- the behaviour is measured, documented, and deliberately not
worked around. Once warm, every request is 70-160ms.
"""

import os

import httpx
import pytest

# A high-risk record: month-to-month, two months in, fibre, no security or support. Chosen
# so the assertions below can be about *content* rather than merely about a 200 -- a
# service returning a plausible-shaped response for the wrong customer would still pass a
# schema-only check.
HIGH_RISK_CUSTOMER = {
    "customerID": "SMOKE-0001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 98.7,
    "TotalCharges": 197.4,
}

# Five attempts at 15s covers one suppressed init plus its re-run with room to spare.
ATTEMPTS = 5
BACKOFF_SECONDS = 15.0
# Longer than API Gateway's own 29s ceiling, so a timeout here means the client gave up
# before the gateway did -- which would be the client's fault, not the service's.
REQUEST_TIMEOUT = 35.0


@pytest.fixture(scope="module")
def base_url() -> str:
    url = os.environ.get("CHURN_SMOKE_URL")
    if not url:
        pytest.skip("CHURN_SMOKE_URL is not set -- smoke tests only run against a live URL")
    return url.rstrip("/")


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=REQUEST_TIMEOUT) as session:
        yield session


def _get(client: httpx.Client, path: str) -> httpx.Response:
    return _retrying(lambda: client.get(path))


def _post(client: httpx.Client, path: str, payload: dict) -> httpx.Response:
    return _retrying(lambda: client.post(path, json=payload))


def _retrying(send) -> httpx.Response:
    """Retry through the cold-start 503 (and the timeout that precedes it).

    A 503 is retried; any other non-200 is returned immediately, because a 422 or a 500 is
    a real answer about a real bug and retrying it only makes the failure slower to read.
    """
    import time

    last: httpx.Response | None = None
    for attempt in range(ATTEMPTS):
        try:
            response = send()
        except httpx.TimeoutException:
            # The cold start can exhaust the gateway's budget before it answers at all.
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(BACKOFF_SECONDS)
            continue

        if response.status_code != 503:
            return response

        last = response
        if attempt < ATTEMPTS - 1:
            time.sleep(BACKOFF_SECONDS)

    assert last is not None
    return last


def test_health_reports_a_model_version(client: httpx.Client) -> None:
    """Readiness, and the deploy-verification hook.

    `/health` depends on the artifact, so `ok` here means the model actually loaded --
    not merely that the function is reachable.
    """
    response = _get(client, "/health")
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"]

    expected = os.environ.get("CHURN_SMOKE_VERSION")
    if expected:
        assert body["model_version"] == expected


def test_score_returns_one_result_per_customer_in_order(client: httpx.Client) -> None:
    """Batch in, batch out, in request order -- the contract `/score` promises."""
    second = dict(
        HIGH_RISK_CUSTOMER,
        customerID="SMOKE-0002",
        tenure=68,
        Contract="Two year",
        InternetService="DSL",
        OnlineSecurity="Yes",
        TechSupport="Yes",
        PaymentMethod="Bank transfer (automatic)",
        MonthlyCharges=45.2,
        TotalCharges=3073.6,
    )
    response = _post(client, "/score", {"customers": [HIGH_RISK_CUSTOMER, second]})
    assert response.status_code == 200, response.text

    body = response.json()
    assert [row["customerID"] for row in body["results"]] == ["SMOKE-0001", "SMOKE-0002"]

    high, low = body["results"]
    assert 0.0 <= high["probability"] <= 1.0
    # Not a metric assertion -- an ordering one. A month-to-month customer two months in
    # must not score below a two-year customer of nearly six years. If that inverts, the
    # feature pipeline is wired wrong in a way no schema check would catch.
    assert high["probability"] > low["probability"]
    assert high["band"] == "high"
    assert low["band"] == "low"


def test_explain_returns_ranked_drivers_for_the_customer(client: httpx.Client) -> None:
    """Drivers come back ranked by magnitude, and describe the customer that was sent."""
    response = _post(client, "/explain", {"customer": HIGH_RISK_CUSTOMER})
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["customerID"] == "SMOKE-0001"
    assert body["band"] == "high"

    drivers = body["drivers"]
    assert 1 <= len(drivers) <= 3
    magnitudes = [abs(driver["contribution"]) for driver in drivers]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert all(driver["direction"] in {"increases", "decreases"} for driver in drivers)


def test_recommend_never_500s_and_stays_inside_the_catalogue(client: httpx.Client) -> None:
    """The route's central guarantee, asserted against the deployment rather than a fake.

    Both outcomes are a pass, and that is the point: with a key the action is one of the
    five, and without one it is `null` with a reason. A 500 is the only real failure, and
    an action outside the catalogue is the one that would matter most -- it would mean an
    offer nobody authorised reached a caller.
    """
    catalogue = {
        "fee_waiver",
        "plan_downgrade",
        "contract_term_incentive",
        "support_callback",
        "loyalty_perk",
    }

    response = _post(client, "/recommend", {"customer": HIGH_RISK_CUSTOMER})
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["customerID"] == "SMOKE-0001"
    assert body["drivers"]

    if body["action"] is None:
        assert body["reason"], "a degraded recommendation must say why"
    else:
        assert body["action"] in catalogue
        assert body["rationale"]
        assert body["reason"] is None

"""The two guarantees `/recommend` makes, and the prompt that has to earn them.

`SPEC.md` §5 asks for exactly two things to be proven: **only catalogue actions escape**,
and **every failure degrades to a served response** rather than a 500. Both are tested
against a fake client -- no network, no key, no recorded fixtures -- because the point is
what this code does with an answer, not what the model would say.

The fake client lives in `fakes.py` because the route tests in `test_api.py` need the
same one -- the degradation contract has to hold at the HTTP edge, not just in the
function.
"""

from types import SimpleNamespace

import anthropic
import httpx
import pytest

from churn.actions import CATALOGUE, ActionId
from churn.explain import Driver
from churn.recommend import (
    EFFORT,
    MAX_RETRIES,
    MODEL,
    TIMEOUT_SECONDS,
    DegradationReason,
    RecommendationDraft,
    build_user_prompt,
    recommend,
)
from churn.scoring import RiskBand
from fakes import FakeClient, draft, responding

DRIVERS = [
    Driver("Contract=Month-to-month", "Month-to-month", 1.42, "increases"),
    Driver("tenure", 3, 0.88, "increases"),
    Driver("InternetService=Fiber optic", "Fiber optic", 0.51, "increases"),
]


# --- guarantee 1: only catalogue actions escape ----------------------------------


def test_a_valid_choice_is_returned_with_its_rationale():
    client = responding(draft())

    result = recommend(client, RiskBand.HIGH, DRIVERS)

    assert result.action is ActionId.CONTRACT_TERM_INCENTIVE
    assert result.rationale == "Because."
    assert result.drivers_used == ("Contract=Month-to-month",)
    assert result.reason is None


@pytest.mark.parametrize("action", [entry.id.value for entry in CATALOGUE])
def test_every_catalogue_action_survives_validation(action):
    """All five, so the catalogue and the validator cannot drift apart silently."""
    result = recommend(responding(draft(action=action)), RiskBand.HIGH, DRIVERS)

    assert result.action == ActionId(action)


@pytest.mark.parametrize(
    "action",
    ["free_iphone", "fee_waiver ", "FEE_WAIVER", "", "cancel_the_account"],
)
def test_an_off_catalogue_action_never_escapes(action):
    """The check our code owns, independent of whether the API honoured its schema."""
    result = recommend(responding(draft(action=action)), RiskBand.HIGH, DRIVERS)

    assert result.action is None
    assert result.reason == DegradationReason.OFF_CATALOGUE
    assert result.rationale is None


def test_a_fabricated_driver_citation_is_rejected():
    """A justification citing evidence we never supplied is worse than no recommendation,
    because it reads as grounded when it is not."""
    client = responding(draft(drivers_used=["PaymentMethod=Electronic check"]))

    result = recommend(client, RiskBand.HIGH, DRIVERS)

    assert result.action is None
    assert result.reason == DegradationReason.UNSUPPORTED_DRIVERS


def test_citing_a_subset_of_the_supplied_drivers_is_fine():
    client = responding(draft(drivers_used=["tenure", "Contract=Month-to-month"]))

    assert recommend(client, RiskBand.HIGH, DRIVERS).action is ActionId.CONTRACT_TERM_INCENTIVE


def test_the_draft_model_itself_accepts_an_off_list_action():
    """Guards the design in `RecommendationDraft`: parsing stays permissive so the
    catalogue check in `_validate` is reachable code rather than dead code.

    If someone "tightens" that field to `ActionId`, this fails and the reason is here.
    """
    parsed = RecommendationDraft(action="nonsense", rationale="x", drivers_used=[])

    assert parsed.action == "nonsense"


def test_the_schema_sent_to_the_api_still_constrains_the_action():
    """The other half of the same design: permissive parsing, constrained request."""
    schema = RecommendationDraft.model_json_schema()

    assert schema["properties"]["action"]["enum"] == [entry.id.value for entry in CATALOGUE]


# --- guarantee 2: every failure degrades to a served response --------------------


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize(
    "failure,expected",
    [
        (anthropic.APITimeoutError(request=_request()), DegradationReason.TIMEOUT),
        (
            anthropic.APIConnectionError(request=_request()),
            DegradationReason.UPSTREAM_ERROR,
        ),
        (
            anthropic.APIStatusError(
                "server error",
                response=httpx.Response(500, request=_request()),
                body=None,
            ),
            DegradationReason.UPSTREAM_ERROR,
        ),
        (RuntimeError("something nobody predicted"), DegradationReason.UPSTREAM_ERROR),
    ],
)
def test_every_call_failure_degrades_instead_of_raising(failure, expected):
    result = recommend(FakeClient(failure), RiskBand.HIGH, DRIVERS)

    assert result.action is None
    assert result.reason == expected


def test_a_missing_api_key_is_a_degraded_response_not_an_error():
    """`None` for the client is the same path as an outage, so it is tested once."""
    result = recommend(None, RiskBand.HIGH, DRIVERS)

    assert result.action is None
    assert result.reason == DegradationReason.NOT_CONFIGURED


def test_a_refusal_degrades():
    client = FakeClient(SimpleNamespace(stop_reason="refusal", parsed_output=None))

    assert recommend(client, RiskBand.HIGH, DRIVERS).reason == DegradationReason.REFUSED


def test_an_unparsed_response_degrades():
    client = FakeClient(SimpleNamespace(stop_reason="end_turn", parsed_output=None))

    assert recommend(client, RiskBand.HIGH, DRIVERS).reason == DegradationReason.MALFORMED


def test_a_dict_shaped_answer_is_validated_rather_than_trusted():
    """A client that hands back a plain dict must not skip the schema."""
    good = responding({"action": "fee_waiver", "rationale": "x", "drivers_used": []})
    bad = responding({"action": "fee_waiver"})

    assert recommend(good, RiskBand.HIGH, DRIVERS).action is ActionId.FEE_WAIVER
    assert recommend(bad, RiskBand.HIGH, DRIVERS).reason == DegradationReason.MALFORMED


# --- the request itself ----------------------------------------------------------


def test_the_call_is_bounded_well_inside_the_lambda_timeout():
    """The 30s function timeout is the budget; these two numbers are what keep the call
    inside it. Asserted because the SDK's defaults (600s, 2 retries) would not."""
    client = responding(draft())

    recommend(client, RiskBand.HIGH, DRIVERS)

    assert client.options == {"timeout": TIMEOUT_SECONDS, "max_retries": MAX_RETRIES}
    assert TIMEOUT_SECONDS * (MAX_RETRIES + 1) < 30


def test_the_request_pins_the_output_schema():
    client = responding(draft())

    recommend(client, RiskBand.HIGH, DRIVERS)

    assert client.messages.calls[0]["output_format"] is RecommendationDraft


# --- the prompt ------------------------------------------------------------------


def test_the_prompt_contains_the_band_the_drivers_and_the_catalogue():
    prompt = build_user_prompt(RiskBand.HIGH, DRIVERS)

    assert "high" in prompt
    for driver in DRIVERS:
        assert driver.feature in prompt
        assert driver.direction in prompt
    for entry in CATALOGUE:
        assert entry.id.value in prompt


def test_the_prompt_carries_no_customer_identifier_or_record():
    """The model gets band, drivers, catalogue -- and nothing else. It cannot invent a
    customer fact it was never given, which is stronger than telling it not to."""
    prompt = build_user_prompt(RiskBand.HIGH, DRIVERS)

    for leak in ["0305-SQECB", "customerID", "MonthlyCharges", "PaymentMethod", "gender"]:
        assert leak not in prompt


def test_the_request_pairs_the_model_with_a_valid_effort_setting():
    """`effort` and the model are one decision, not two.

    Sonnet 5 accepts `effort` and needs it -- omitting it means adaptive thinking at the
    default `high`, against an 8s budget. The Haiku 4.5 family rejects it with a 400. So a
    model swap that leaves this line alone breaks only against the live API, where the
    failure surfaces as a permanently degraded recommendation rather than a test error.
    """
    client = responding(draft())

    recommend(client, RiskBand.HIGH, DRIVERS)

    call = client.messages.calls[0]
    assert call["model"] == MODEL
    if MODEL.startswith("claude-haiku"):
        assert "output_config" not in call, "Haiku rejects output_config.effort with a 400"
    else:
        assert call["output_config"] == {"effort": EFFORT}

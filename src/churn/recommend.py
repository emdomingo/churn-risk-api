"""The constrained LLM layer: drivers in, one catalogue action out.

**What the model is allowed to see.** Risk band, the top three drivers, and the
catalogue. Nothing else -- no customer id, no raw record, no free-text notes. It cannot
invent a customer fact because it was never given one, which is a stronger guarantee than
instructing it not to. Everything it does see is already in the `/explain` response, so
the prompt adds no new exposure over an endpoint the caller can already reach.

**Two independent constraints on the answer, deliberately.** The request pins
`output_config.format` to a JSON schema whose `action` is an enum of the five ids, so the
API itself will not emit anything else. Then `_validate` checks the answer against
`actions.BY_ID` anyway. The belt-and-braces is the point: the schema is the API's promise
and the validator is ours, and `SPEC.md` §5 asks for a check we own. If the schema layer
ever changes, the guarantee this service makes does not depend on it.

**Nothing here raises.** `/recommend` must never 500 because an external service is slow
or down, so every failure -- timeout, HTTP error, refusal, malformed answer, an action
off the catalogue, a driver the model made up -- resolves to a `Recommendation` with
`action=None` and a `reason` naming what happened. The caller still gets its score and
drivers. This is the one module in the codebase where a bare `except Exception` is
correct, and it is guarded by a `reason` that says which branch was taken.

**Gate: model, timeout and retries.** All three are set here rather than in `config`
because they are engineering limits, not deployment knobs -- see the constants for the
arithmetic against the 29s Lambda budget.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum

import anthropic
from pydantic import BaseModel, Field, ValidationError

from churn.actions import CATALOGUE, ActionId, catalogue_for_prompt, is_in_catalogue
from churn.explain import Driver
from churn.scoring import RiskBand

logger = logging.getLogger("churn.recommend")

# Owner's call (S5). The task is a five-way selection over three structured inputs with
# the answer shape pinned by a JSON schema, and the part that has to be *right* -- only
# catalogue actions escape, every cited driver was supplied -- is enforced by `_validate`
# rather than by the model's judgement. What the larger model buys is the quality of the
# *rationale*, which is the part a human reads. Latency is the constraint that bounds the
# choice: this call sits inside a 29s Lambda budget behind a cold start.
MODEL = "claude-sonnet-5"

# Gate: LLM timeout and retries, resolved (S5). The Lambda budget is 29s -- API Gateway's
# HTTP API ceiling, adopted as the function timeout in S8. Scoring and explaining one
# customer costs ~50ms warm, so effectively the whole budget belongs to this call. The SDK
# retries timeouts, so worst-case wall clock is `TIMEOUT_SECONDS * (MAX_RETRIES + 1)` =
# 16s, which leaves ~13s for a measured ~6s cold-start init. The SDK's own defaults
# (600s timeout, 2 retries) would blow the function timeout and turn a slow LLM into a
# 502 from API Gateway rather than the graceful degradation below.
TIMEOUT_SECONDS = 8.0
MAX_RETRIES = 1

# A rationale is one or two sentences; the schema caps it and this caps the bill.
MAX_TOKENS = 1024

# Sonnet 5 runs adaptive thinking whenever `thinking` is omitted, so depth is controlled
# through `effort` instead -- and it has to be, because the default is `high` and this
# call has an 8s budget. Low is the honest setting for the work: pick one of five, cite
# the drivers, write two sentences. This is also the line that changes if a model swap
# ever happens again -- `effort` is rejected outright by the Haiku 4.5 family, so it
# travels with the model choice above rather than being an independent knob.
EFFORT = "low"

SYSTEM_PROMPT = """You are a retention analyst. You select exactly one retention action \
from a fixed catalogue for a customer at risk of churning.

Rules:
- Choose only from the catalogue you are given. Never propose anything else.
- Reason only from the risk band and drivers you are given. You know nothing else about \
this customer -- do not assume or invent tenure, spend, history, or contact details.
- Justify the choice by referring to the drivers. Cite only drivers from the list.
- A driver marked "decreases" is protecting the customer, not endangering them. Do not \
recommend an action that addresses something already in the customer's favour.
- Keep the rationale to one or two sentences a retention agent could act on."""


class DegradationReason(StrEnum):
    """Why there is no action. Values are stable strings -- they reach the API response."""

    NOT_CONFIGURED = "llm_not_configured"
    TIMEOUT = "llm_timeout"
    UPSTREAM_ERROR = "llm_upstream_error"
    REFUSED = "llm_refused"
    MALFORMED = "llm_malformed_response"
    OFF_CATALOGUE = "action_not_in_catalogue"
    UNSUPPORTED_DRIVERS = "drivers_not_in_evidence"


class RecommendationDraft(BaseModel):
    """The model's answer, before we decide whether to trust it.

    Separate from the response model in `schemas` on purpose: this is what an untrusted
    party produced, and it earns its way into a response only by passing `_validate`.

    **`action` is a `str`, not an `ActionId`, and that is the whole design.** Typing it as
    the enum would make Pydantic reject an off-list value here -- which sounds stricter
    but is actually weaker: the catalogue check in `_validate` would become unreachable
    code, and the guarantee this service makes would rest entirely on the API honouring
    its own schema. `json_schema_extra` still puts the enum in the schema the API
    receives, so the constraint is sent; parsing stays permissive so our own check is the
    one that decides. Two layers, only one of which we control, and the one we control is
    the one that is tested.
    """

    action: str = Field(json_schema_extra={"enum": [entry.id.value for entry in CATALOGUE]})
    rationale: str = Field(min_length=1, max_length=600)
    drivers_used: list[str]


@dataclass(frozen=True)
class Recommendation:
    """Either an action with its justification, or a reason there is none. Never both."""

    action: ActionId | None
    rationale: str | None
    drivers_used: tuple[str, ...]
    reason: str | None


def _degraded(reason: DegradationReason) -> Recommendation:
    return Recommendation(action=None, rationale=None, drivers_used=(), reason=reason.value)


def build_user_prompt(band: RiskBand, drivers: list[Driver]) -> str:
    """The entire customer-specific half of the prompt. Pure, so it is testable alone.

    Driver values are included -- `Contract=Two year` with the customer's own value is
    what tells the model a contract-term incentive is pointless here. They are facts the
    model was handed, not facts it can invent, which is the distinction that matters.
    """
    lines = [
        f"Risk band: {band.value}",
        "",
        "Top drivers (most influential first):",
    ]
    for driver in drivers:
        lines.append(
            f"- {driver.feature} (customer's value: {driver.value}) "
            f"{driver.direction} churn risk, magnitude {abs(driver.contribution):.2f}"
        )
    lines += ["", "Allowed actions:", catalogue_for_prompt()]
    return "\n".join(lines)


def _validate(draft: RecommendationDraft, drivers: list[Driver]) -> Recommendation:
    """Decide whether the model's answer is usable. The auditability guarantee lives here.

    Two checks. The action must name a catalogue entry -- the schema should already
    guarantee it, and we check anyway. And every cited driver must be one we supplied: a
    citation we did not provide is a fabricated justification, which is worse than no
    recommendation, because it reads as evidence.
    """
    if not is_in_catalogue(draft.action):
        logger.warning("off-catalogue action rejected")
        return _degraded(DegradationReason.OFF_CATALOGUE)

    supplied = {driver.feature for driver in drivers}
    if not set(draft.drivers_used) <= supplied:
        logger.warning("fabricated driver citation rejected")
        return _degraded(DegradationReason.UNSUPPORTED_DRIVERS)

    return Recommendation(
        action=ActionId(draft.action),
        rationale=draft.rationale,
        drivers_used=tuple(draft.drivers_used),
        reason=None,
    )


def recommend(
    client: anthropic.Anthropic | None, band: RiskBand, drivers: list[Driver]
) -> Recommendation:
    """One catalogue action for this customer, or a reason there is none.

    `client` is injected rather than constructed so the tests never touch the network and
    so a missing API key is an ordinary degraded response instead of a startup failure.
    """
    if client is None:
        return _degraded(DegradationReason.NOT_CONFIGURED)

    try:
        response = client.with_options(
            timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES
        ).messages.parse(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            output_config={"effort": EFFORT},
            output_format=RecommendationDraft,
            messages=[{"role": "user", "content": build_user_prompt(band, drivers)}],
        )
    except anthropic.APITimeoutError:
        logger.warning("llm timeout", extra={"error": "APITimeoutError"})
        return _degraded(DegradationReason.TIMEOUT)
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
        logger.warning("llm unavailable", extra={"error": type(exc).__name__})
        return _degraded(DegradationReason.UPSTREAM_ERROR)
    except ValidationError:
        # The SDK parses into `RecommendationDraft` itself, so a response that does not
        # fit the schema fails here rather than below.
        logger.warning("llm response did not match the schema")
        return _degraded(DegradationReason.MALFORMED)
    except Exception as exc:  # noqa: BLE001 - see the module docstring
        logger.warning("llm call failed", extra={"error": type(exc).__name__})
        return _degraded(DegradationReason.UPSTREAM_ERROR)

    if getattr(response, "stop_reason", None) == "refusal":
        return _degraded(DegradationReason.REFUSED)

    draft = getattr(response, "parsed_output", None)
    if draft is None:
        return _degraded(DegradationReason.MALFORMED)

    if not isinstance(draft, RecommendationDraft):
        # A mocked or future client could hand back a plain dict; validate rather than
        # trust the type, since this is the untrusted boundary.
        try:
            draft = RecommendationDraft.model_validate(draft)
        except (ValidationError, TypeError):
            return _degraded(DegradationReason.MALFORMED)

    return _validate(draft, drivers)


# Re-exported so callers can enumerate the catalogue without a second import.
__all__ = [
    "CATALOGUE",
    "ActionId",
    "DegradationReason",
    "Recommendation",
    "RecommendationDraft",
    "build_user_prompt",
    "recommend",
]

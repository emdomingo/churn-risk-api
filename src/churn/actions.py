"""The closed catalogue of retention actions.

**Why a catalogue rather than free text** (`SPEC.md` §5 names this as a gate): a retention
offer is a commitment with a cost attached. An LLM writing prose could invent "three
months free", promise a discount nobody approved, or produce something a retention agent
has no way to execute. Restricting it to five identifiers turns an open-ended generation
problem into a selection problem: the model's whole job is to pick one and say why. The
consequences are worth stating plainly --

- **Auditable.** Every recommendation is one of five known values, so "what did we offer
  and why" is answerable from a log line rather than by reading generated prose.
- **Bounded cost.** Finance signs off on five offers once, not on whatever the model
  produced this morning.
- **Testable.** `test_recommend.py` can assert that nothing outside this tuple ever
  escapes -- an assertion that is impossible to write against free text.
- **Degradable.** An off-list answer is detectable, so the failure mode is "no action
  returned" rather than "an action nobody authorised".

The identifiers are what the API returns and what gets logged; the labels and
descriptions are what the model reads. Both live here so the prompt and the validator
cannot drift apart -- the same tuple builds the prompt and checks the answer.
"""

from dataclasses import dataclass
from enum import StrEnum


class ActionId(StrEnum):
    """A `StrEnum` so it serialises as a plain string and compares as one."""

    FEE_WAIVER = "fee_waiver"
    PLAN_DOWNGRADE = "plan_downgrade"
    CONTRACT_TERM_INCENTIVE = "contract_term_incentive"
    SUPPORT_CALLBACK = "support_callback"
    LOYALTY_PERK = "loyalty_perk"


@dataclass(frozen=True)
class CatalogueAction:
    id: ActionId
    label: str
    description: str


# The five from `SPEC.md` §5, unchanged. Descriptions are written for the model: each one
# says what the action *is* and what kind of driver makes it the right pick, because the
# model is choosing between them on the strength of the drivers alone.
CATALOGUE: tuple[CatalogueAction, ...] = (
    CatalogueAction(
        ActionId.FEE_WAIVER,
        "Fee waiver",
        "Waive an upcoming fee or surcharge for one billing cycle. Fits a customer whose "
        "risk is driven by what they pay rather than by what they receive.",
    ),
    CatalogueAction(
        ActionId.PLAN_DOWNGRADE,
        "Plan downgrade offer",
        "Offer a smaller plan at a lower price. Fits high charges paired with services "
        "the customer is not using -- keeps the relationship at a price that fits it.",
    ),
    CatalogueAction(
        ActionId.CONTRACT_TERM_INCENTIVE,
        "Contract-term incentive",
        "Offer a discount in exchange for committing to a one- or two-year term. Fits a "
        "month-to-month customer; pointless for one already on a longer term.",
    ),
    CatalogueAction(
        ActionId.SUPPORT_CALLBACK,
        "Support callback",
        "Schedule a proactive call from technical support. Fits risk driven by missing "
        "support or protection services rather than by price.",
    ),
    CatalogueAction(
        ActionId.LOYALTY_PERK,
        "Loyalty perk",
        "Grant a complimentary add-on or service credit recognising tenure. Fits a "
        "long-standing customer with no single dominant price or service driver.",
    ),
)

BY_ID: dict[str, CatalogueAction] = {action.id.value: action for action in CATALOGUE}


def is_in_catalogue(value: object) -> bool:
    """Whether a model-supplied value names a real action.

    Takes `object` rather than `str` on purpose: this guards the boundary where a
    response has not been trusted yet, and `None` or a number must answer `False`
    instead of raising.
    """
    return isinstance(value, str) and value in BY_ID


def catalogue_for_prompt() -> str:
    """The catalogue as the model sees it: one line per action, id first.

    Built from `CATALOGUE` rather than written out as a prompt string, so adding an
    action cannot leave the prompt and the validator disagreeing about what is allowed.
    """
    return "\n".join(
        f"- {action.id.value}: {action.label}. {action.description}" for action in CATALOGUE
    )

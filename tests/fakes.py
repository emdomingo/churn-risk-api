"""A stand-in for `anthropic.Anthropic`, shared by the unit and route tests.

Deliberately dumb: it records the kwargs it was called with and returns whatever it was
constructed with, raising it instead if that happens to be an exception. Anything
smarter would start testing the SDK rather than our handling of it.

Importable as `from fakes import ...` because `tests/` is not a package -- pytest puts
the test directory itself on `sys.path`, so `tests.fakes` would not resolve.
"""

from types import SimpleNamespace

from churn.recommend import RecommendationDraft


class FakeMessages:
    def __init__(self, result):
        self._result = result
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeClient:
    def __init__(self, result):
        self.messages = FakeMessages(result)
        self.options: dict | None = None

    def with_options(self, **kwargs):
        self.options = kwargs
        return self


def draft(action="contract_term_incentive", drivers_used=None, rationale="Because."):
    """A well-formed answer from the model. `action` is a `str`, as the real one is."""
    return RecommendationDraft(
        action=action,
        rationale=rationale,
        drivers_used=["Contract=Month-to-month"] if drivers_used is None else drivers_used,
    )


def responding(result):
    """A client whose `parse` returns a successful, non-refused response."""
    return FakeClient(SimpleNamespace(stop_reason="end_turn", parsed_output=result))


def failing(exception):
    """A client whose `parse` raises."""
    return FakeClient(exception)

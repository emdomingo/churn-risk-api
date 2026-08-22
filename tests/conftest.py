"""Shared fixtures.

The artifact is never committed (`artifacts/` is gitignored), so the suite cannot load a
prebuilt model — it trains its own, once per session, on the 200-row fixture. That is what
lets CI run with no AWS, no network, and no model file, and it means the tests exercise
the same `build_artifact` code path that produces the real artifact rather than a
test-only imitation of it.

Scoped to the session because the fit costs a second or two and nothing mutates it.
"""

from pathlib import Path

import pandas as pd
import pytest

from churn.artifact import ChurnArtifact
from churn.data import clean, load_raw
from churn.train import build_artifact

FIXTURE = Path(__file__).parent / "fixtures" / "telco_sample.csv"


@pytest.fixture(scope="session")
def sample_frame() -> pd.DataFrame:
    """The 200-row fixture, cleaned — the input `build_artifact` expects."""
    return clean(load_raw(FIXTURE))


@pytest.fixture(scope="session")
def tiny_artifact(sample_frame: pd.DataFrame) -> ChurnArtifact:
    """A real artifact trained on the fixture. Too small to be accurate; that is fine —
    every test here is about wiring and contracts, never about model quality."""
    return build_artifact(sample_frame, model_version="test-artifact")

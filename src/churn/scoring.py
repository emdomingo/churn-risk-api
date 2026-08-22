"""Record in, calibrated probability and risk band out.

Pure functions over a fitted `ChurnArtifact`: no web framework, no AWS, no file IO. The
API layer's whole job on `/score` is to validate JSON into records, call `score`, and
serialise the result.

**The bands are not cosmetic thresholds.** `LOW_BAND_MAX` is the same number as the
cost-minimising operating threshold in `train.operating_threshold` — `1 / (1 + 5.0)` for
the assumed cost ratio — so "the band is not low" and "the model says intervene" are the
same predicate, and the confusion matrix in the metrics table describes exactly the
medium+high population. `HIGH_BAND_MIN` is a second, stricter cut chosen from the test
split's precision curve: at 0.50 the high band is 23% of customers and 63% of them churn,
against a 26.5% base rate. Medium means "worth a cheap touch", high means "worth spending
real money on".

The constants live here rather than being imported from `train`, because `train` is
offline code that the Lambda image has no reason to import. `test_scoring.py` asserts the
low cut still equals `operating_threshold()`, so the two cannot drift apart silently.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from churn.artifact import ChurnArtifact

# Gate #4, resolved. See the module docstring for where each number comes from.
LOW_BAND_MAX = 1.0 / 6.0
HIGH_BAND_MIN = 0.50


class RiskBand(StrEnum):
    """A `StrEnum` so it serialises to a plain JSON string with no encoder help."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class Score:
    probability: float
    band: RiskBand


def risk_band(probability: float) -> RiskBand:
    """Map a calibrated probability onto a band. Boundaries belong to the upper band."""
    if probability < LOW_BAND_MAX:
        return RiskBand.LOW
    if probability < HIGH_BAND_MIN:
        return RiskBand.MEDIUM
    return RiskBand.HIGH


def as_frame(records: Mapping | Sequence[Mapping] | pd.DataFrame) -> pd.DataFrame:
    """Accept one record, many records, or a frame — always return a frame.

    Scoring one customer and scoring a batch are the same code path underneath, which is
    what keeps `/score` (batch) and `/explain` (single) from drifting apart.
    """
    if isinstance(records, pd.DataFrame):
        return records
    if isinstance(records, Mapping):
        return pd.DataFrame([records])
    return pd.DataFrame(list(records))


def score(
    artifact: ChurnArtifact, records: Mapping | Sequence[Mapping] | pd.DataFrame
) -> list[Score]:
    """Calibrated churn probability and band for each record, in input order.

    One `transform` and one `predict_proba` for the whole batch: both are vectorised, so
    a 100-record request costs barely more than a single one. Column 1 is the positive
    class because `clean` maps `Churn` to 0/1.
    """
    frame = as_frame(records)
    probabilities = artifact.calibrator.predict_proba(artifact.transform(frame))[:, 1]
    return [Score(probability=float(p), band=risk_band(float(p))) for p in probabilities]


def score_one(artifact: ChurnArtifact, record: Mapping | pd.DataFrame) -> Score:
    """Single-record convenience for `/explain` and `/recommend`."""
    return score(artifact, record)[0]

"""Tests for the scoring core.

Two kinds of thing are worth pinning here. The band boundaries are a *decision*, so they
are tested at their exact edges — an off-by-one on `<` vs `<=` silently moves customers
between "leave alone" and "spend money on". Everything else guards the contract the API
layer will rely on: probabilities really are probabilities, and batch scoring is the same
computation as single scoring.
"""

import pandas as pd
import pytest

from churn.artifact import ChurnArtifact
from churn.data import TARGET
from churn.features import split_xy
from churn.scoring import (
    HIGH_BAND_MIN,
    LOW_BAND_MAX,
    RiskBand,
    as_frame,
    risk_band,
    score,
    score_one,
)
from churn.train import operating_threshold


@pytest.fixture(scope="module")
def records(sample_frame: pd.DataFrame) -> pd.DataFrame:
    X, _ = split_xy(sample_frame)
    return X


def test_low_band_cut_is_the_cost_derived_operating_threshold() -> None:
    """The bands and the confusion matrix must describe the same population.

    If this fails, someone changed `MISSED_CHURNER_COST_RATIO` or nudged a band without
    noticing that "not low" is supposed to mean "the model says intervene".
    """
    assert LOW_BAND_MAX == pytest.approx(operating_threshold())


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, RiskBand.LOW),
        (LOW_BAND_MAX - 1e-9, RiskBand.LOW),
        (LOW_BAND_MAX, RiskBand.MEDIUM),
        (HIGH_BAND_MIN - 1e-9, RiskBand.MEDIUM),
        (HIGH_BAND_MIN, RiskBand.HIGH),
        (1.0, RiskBand.HIGH),
    ],
)
def test_band_boundaries_belong_to_the_upper_band(probability: float, expected: RiskBand) -> None:
    assert risk_band(probability) is expected


def test_bands_serialise_as_plain_strings() -> None:
    """`/score` returns JSON; a `StrEnum` member must not leak as 'RiskBand.HIGH'."""
    assert risk_band(0.9) == "high"
    assert f"{risk_band(0.9)}" == "high"


def test_scores_are_probabilities(tiny_artifact: ChurnArtifact, records: pd.DataFrame) -> None:
    for scored in score(tiny_artifact, records):
        assert 0.0 <= scored.probability <= 1.0
        assert isinstance(scored.probability, float)


def test_band_always_matches_its_own_probability(
    tiny_artifact: ChurnArtifact, records: pd.DataFrame
) -> None:
    for scored in score(tiny_artifact, records):
        assert scored.band is risk_band(scored.probability)


def test_batch_scoring_preserves_input_order(
    tiny_artifact: ChurnArtifact, records: pd.DataFrame
) -> None:
    """Batch is list-in/list-out; row i of the response must be record i of the request."""
    batch = score(tiny_artifact, records.head(10))
    individually = [score_one(tiny_artifact, records.iloc[[i]]) for i in range(10)]
    assert [s.probability for s in batch] == pytest.approx(
        [s.probability for s in individually]
    )


def test_a_single_dict_record_scores(tiny_artifact: ChurnArtifact, records: pd.DataFrame) -> None:
    """This is the shape FastAPI will hand over: one JSON object, already validated."""
    record = records.iloc[0].to_dict()
    assert 0.0 <= score_one(tiny_artifact, record).probability <= 1.0


def test_as_frame_accepts_one_record_many_records_and_a_frame(records: pd.DataFrame) -> None:
    one = records.iloc[0].to_dict()
    many = [records.iloc[i].to_dict() for i in range(3)]
    assert as_frame(one).shape == (1, records.shape[1])
    assert as_frame(many).shape == (3, records.shape[1])
    assert as_frame(records) is records


def test_scoring_ignores_an_unexpected_extra_column(
    tiny_artifact: ChurnArtifact, records: pd.DataFrame
) -> None:
    """`remainder="drop"` means a stray field cannot change the score."""
    record = records.iloc[[0]]
    noisy = record.assign(favourite_colour="blue")
    assert score_one(tiny_artifact, noisy).probability == pytest.approx(
        score_one(tiny_artifact, record).probability
    )


def test_churners_score_higher_on_average_than_non_churners(
    tiny_artifact: ChurnArtifact, sample_frame: pd.DataFrame
) -> None:
    """A sanity check on wiring, not on accuracy: if the label columns were swapped or
    the wrong `predict_proba` column taken, this is what would catch it."""
    X, y = split_xy(sample_frame)
    probabilities = pd.Series([s.probability for s in score(tiny_artifact, X)], index=X.index)
    assert probabilities[y == 1].mean() > probabilities[y == 0].mean()


def test_target_column_is_not_required_at_inference(
    tiny_artifact: ChurnArtifact, sample_frame: pd.DataFrame
) -> None:
    """Real requests never carry the answer."""
    assert TARGET not in split_xy(sample_frame)[0].columns

"""Tests for the explanation core.

Split in two on purpose. `select_drivers` is *policy* — which contributions a caller
sees — so it is tested against hand-built Series with no artifact, no SHAP, and no model
in sight; that is the whole reason it takes a Series rather than a record. The rest is
*plumbing*: names reconstructed from the encoder, signs, and the shape of the `Driver`
objects `/explain` and `/recommend` will consume.
"""

import pandas as pd
import pytest

from churn.artifact import ChurnArtifact
from churn.data import TARGET
from churn.explain import (
    RELATIVE_NOISE_FLOOR,
    Driver,
    contributions,
    driver_name,
    explain,
    select_drivers,
    source_columns,
)
from churn.features import NUMERIC_FEATURES, split_xy


@pytest.fixture(scope="module")
def records(sample_frame: pd.DataFrame) -> pd.DataFrame:
    X, _ = split_xy(sample_frame)
    return X


@pytest.fixture(scope="module")
def mapping(tiny_artifact: ChurnArtifact) -> dict[str, tuple[str, str | None]]:
    return source_columns(tiny_artifact.preprocessor)


@pytest.fixture(scope="module")
def riskiest_record(tiny_artifact: ChurnArtifact, records: pd.DataFrame) -> pd.DataFrame:
    """The highest-scoring row in the fixture — the record whose drivers must read as
    reasons to *leave*, which is PLAN's done-when criterion for S3."""
    probabilities = tiny_artifact.calibrator.predict_proba(tiny_artifact.transform(records))[:, 1]
    return records.iloc[[int(probabilities.argmax())]]


# --- policy: select_drivers, no model involved --------------------------------------


def test_ranks_by_magnitude_not_by_signed_value() -> None:
    """A strong protective factor outranks a weak risk factor. This is the decision:
    reporting three feeble positives for a safe customer would invent a risk story."""
    values = pd.Series({"tenure": -1.4, "gender_Female": 0.15, "MonthlyCharges": -0.9})
    assert list(select_drivers(values, top_n=2).index) == ["tenure", "MonthlyCharges"]


def test_returns_signed_values_unmodified() -> None:
    """`explain` reads the sign to set `direction`; returning magnitudes would label
    every driver as 'increases'."""
    values = pd.Series({"tenure": -1.4, "Contract_Month-to-month": 0.8})
    assert select_drivers(values, top_n=2).to_dict() == {
        "tenure": -1.4,
        "Contract_Month-to-month": 0.8,
    }


def test_returns_encoded_names_not_display_names() -> None:
    """`explain` looks the index up in `source_columns`, which is keyed on the encoder's
    own names — renaming here would break that lookup."""
    values = pd.Series({"Contract_Two year": -0.9, "tenure": 0.4})
    assert list(select_drivers(values, top_n=1).index) == ["Contract_Two year"]


def test_never_returns_more_than_top_n() -> None:
    values = pd.Series({f"f{i}": 1.0 - i / 100 for i in range(46)})
    assert len(select_drivers(values, top_n=3)) == 3


def test_noise_floor_is_relative_to_the_strongest_driver() -> None:
    """Same absolute contribution, two customers, two verdicts: 0.03 is noise beside a
    2.0 driver and signal beside a 0.1 one. A fixed cutoff cannot express that."""
    dominated = pd.Series({"tenure": 2.0, "MonthlyCharges": 0.5, "gender_Female": 0.03})
    flat = pd.Series({"tenure": 0.10, "MonthlyCharges": 0.05, "gender_Female": 0.03})

    assert list(select_drivers(dominated, top_n=3).index) == ["tenure", "MonthlyCharges"]
    assert len(select_drivers(flat, top_n=3)) == 3


def test_floor_boundary_survives() -> None:
    """A contribution exactly at the floor is kept, not dropped."""
    values = pd.Series({"tenure": 1.0, "gender_Female": RELATIVE_NOISE_FLOOR})
    assert len(select_drivers(values, top_n=3)) == 2


def test_returns_fewer_than_top_n_rather_than_padding_with_noise() -> None:
    """The whole point of the floor: two real drivers beats two real drivers plus a
    fabricated third that `/recommend` would have to justify."""
    values = pd.Series({"tenure": 1.0, "MonthlyCharges": 0.8, "gender_Female": 0.001})
    assert len(select_drivers(values, top_n=3)) == 2


def test_all_zero_contributions_return_nothing() -> None:
    """A flat customer must return nothing, not `top_n` drivers of magnitude 0.0.

    Without the guard the floor itself is `0.05 * 0.0 == 0.0`, which every feature
    clears — the noise filter would wave through pure noise.
    """
    assert select_drivers(pd.Series({"tenure": 0.0, "MonthlyCharges": 0.0})).empty


def test_missing_contributions_return_nothing() -> None:
    """NaN fails every comparison, so `strongest == 0` would let it through and rank the
    features by a filter that is silently False everywhere. `not strongest > 0` catches
    it, as does an empty Series, whose `max()` is also NaN."""
    assert select_drivers(pd.Series({"tenure": float("nan")})).empty
    assert select_drivers(pd.Series([], dtype="float64")).empty


def test_selection_is_deterministic_under_ties() -> None:
    """Identical customers must get identical responses, not sort-order roulette."""
    values = pd.Series({"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5})
    assert list(select_drivers(values, top_n=2).index) == list(
        select_drivers(values.copy(), top_n=2).index
    )


# --- plumbing: names, contributions, and the assembled Driver -----------------------


def test_every_encoded_column_maps_back_to_a_source_column(
    tiny_artifact: ChurnArtifact, mapping: dict
) -> None:
    assert set(mapping) == set(tiny_artifact.feature_names)


def test_numeric_columns_map_to_themselves_with_no_level(mapping: dict) -> None:
    for column in NUMERIC_FEATURES:
        assert mapping[column] == (column, None)


def test_driver_names_are_level_specific(mapping: dict) -> None:
    """Gate #3: `/recommend` needs the level to pick an action. 'Contract' alone cannot
    tell it whether a contract-term incentive is the right offer."""
    assert driver_name("Contract_Month-to-month", mapping) == "Contract=Month-to-month"
    assert driver_name("tenure", mapping) == "tenure"


def test_driver_names_survive_levels_containing_punctuation(mapping: dict) -> None:
    """`PaymentMethod_Bank transfer (automatic)` is reconstructed from the encoder's
    categories, not by splitting the string — spaces and parens are just data."""
    assert (
        driver_name("PaymentMethod_Bank transfer (automatic)", mapping)
        == "PaymentMethod=Bank transfer (automatic)"
    )


def test_contributions_cover_every_feature(
    tiny_artifact: ChurnArtifact, riskiest_record: pd.DataFrame
) -> None:
    values = contributions(tiny_artifact, riskiest_record)
    assert list(values.index) == tiny_artifact.feature_names
    assert values.dtype == "float64"


def test_contributions_reconstruct_the_margin(
    tiny_artifact: ChurnArtifact, riskiest_record: pd.DataFrame
) -> None:
    """SHAP additivity, against the *booster* — the margin the drivers actually explain,
    not the calibrated score reported beside them."""
    encoded = tiny_artifact.transform(riskiest_record)
    margin = float(tiny_artifact.booster.predict(encoded, output_margin=True)[0])
    total = contributions(tiny_artifact, riskiest_record).sum()
    assert total + float(tiny_artifact.explainer.expected_value) == pytest.approx(margin, abs=1e-4)


def test_explain_rejects_a_batch(tiny_artifact: ChurnArtifact, records: pd.DataFrame) -> None:
    """`/explain` is single-record by contract; silently explaining row 0 would be worse."""
    with pytest.raises(ValueError, match="exactly one record"):
        explain(tiny_artifact, records.head(3))


def test_riskiest_customer_gets_risk_increasing_drivers(
    tiny_artifact: ChurnArtifact, riskiest_record: pd.DataFrame
) -> None:
    """PLAN's done-when for S3: correct signs on a known-risky record."""
    drivers = explain(tiny_artifact, riskiest_record)
    assert drivers
    assert drivers[0].direction == "increases"
    assert drivers[0].contribution > 0


def test_drivers_are_ordered_strongest_first(
    tiny_artifact: ChurnArtifact, riskiest_record: pd.DataFrame
) -> None:
    magnitudes = [abs(d.contribution) for d in explain(tiny_artifact, riskiest_record)]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_direction_always_agrees_with_the_sign(
    tiny_artifact: ChurnArtifact, records: pd.DataFrame
) -> None:
    for i in range(20):
        for driver in explain(tiny_artifact, records.iloc[[i]]):
            expected = "increases" if driver.contribution > 0 else "decreases"
            assert driver.direction == expected


def test_driver_reports_the_raw_value_of_its_source_column(
    tiny_artifact: ChurnArtifact, riskiest_record: pd.DataFrame
) -> None:
    """Unscaled numerics pay off here: tenure comes back in months, not in sigmas."""
    for driver in explain(tiny_artifact, riskiest_record):
        column = driver.feature.split("=")[0]
        assert driver.value == riskiest_record.iloc[0][column]


def test_explain_accepts_a_plain_dict(
    tiny_artifact: ChurnArtifact, riskiest_record: pd.DataFrame
) -> None:
    """The shape FastAPI will hand over."""
    drivers = explain(tiny_artifact, riskiest_record.iloc[0].to_dict())
    assert all(isinstance(d, Driver) for d in drivers)


def test_explain_never_names_the_target(
    tiny_artifact: ChurnArtifact, records: pd.DataFrame
) -> None:
    for i in range(10):
        assert all(TARGET not in d.feature for d in explain(tiny_artifact, records.iloc[[i]]))

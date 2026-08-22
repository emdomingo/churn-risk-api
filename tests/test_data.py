"""Tests for the data layer.

These run against the committed 200-row fixture, never the full dataset and never the
network, so they stay fast and CI stays self-contained.
"""

from pathlib import Path

import pandas as pd
import pytest

from churn.data import ID_COLUMN, TARGET, Splits, clean, load_raw, split

FIXTURE = Path(__file__).parent / "fixtures" / "telco_sample.csv"


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return load_raw(FIXTURE)


@pytest.fixture(scope="module")
def cleaned(raw: pd.DataFrame) -> pd.DataFrame:
    return clean(raw)


# --- fixture integrity -------------------------------------------------------------
# If these fail the fixture has been regenerated badly and every test below is testing
# nothing. Regenerate with: uv run python scripts/make_fixture.py


def test_fixture_contains_literal_blank_total_charges(raw: pd.DataFrame) -> None:
    """The blanks must survive as ' ' strings, not as NaN, or the coercion path in
    `clean` is never exercised."""
    blanks = raw["TotalCharges"].astype(str).str.strip().eq("")
    assert blanks.sum() == 11
    assert not raw.loc[blanks, "TotalCharges"].isna().any()


def test_fixture_blank_rows_are_all_zero_tenure(raw: pd.DataFrame) -> None:
    blanks = raw["TotalCharges"].astype(str).str.strip().eq("")
    assert (raw.loc[blanks, "tenure"] == 0).all()


# --- clean -------------------------------------------------------------------------


def test_total_charges_is_float_with_no_nan(cleaned: pd.DataFrame) -> None:
    assert cleaned["TotalCharges"].dtype == "float64"
    assert not cleaned["TotalCharges"].isna().any()


def test_blank_total_charges_becomes_zero(raw: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    blanks = raw["TotalCharges"].astype(str).str.strip().eq("")
    assert (cleaned.loc[blanks.to_numpy(), "TotalCharges"] == 0.0).all()


def test_no_rows_are_silently_dropped(raw: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    assert len(cleaned) == len(raw)


def test_id_column_is_dropped(cleaned: pd.DataFrame) -> None:
    assert ID_COLUMN not in cleaned.columns


def test_target_is_binary_int(cleaned: pd.DataFrame) -> None:
    assert cleaned[TARGET].dtype == "int64"
    assert set(cleaned[TARGET].unique()) <= {0, 1}


def test_senior_citizen_normalised_to_yes_no(cleaned: pd.DataFrame) -> None:
    assert set(cleaned["SeniorCitizen"].unique()) <= {"Yes", "No"}


def test_no_nans_anywhere(cleaned: pd.DataFrame) -> None:
    assert not cleaned.isna().any().any()


def test_clean_does_not_mutate_its_input(raw: pd.DataFrame) -> None:
    before = raw.copy()
    clean(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_clean_on_a_hand_built_frame() -> None:
    """Unit-level check with no file involved."""
    df = pd.DataFrame(
        {
            ID_COLUMN: ["0001-AAA", "0002-BBB"],
            "SeniorCitizen": [0, 1],
            "tenure": [0, 12],
            "MonthlyCharges": [50.0, 70.0],
            "TotalCharges": [" ", "840"],
            TARGET: ["No", "Yes"],
        }
    )

    out = clean(df)

    assert ID_COLUMN not in out.columns
    assert out["TotalCharges"].tolist() == [0.0, 840.0]
    assert out["SeniorCitizen"].tolist() == ["No", "Yes"]
    assert out[TARGET].tolist() == [0, 1]


# --- split -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def splits(cleaned: pd.DataFrame) -> Splits:
    return split(cleaned)


def test_split_is_exhaustive_and_disjoint(cleaned: pd.DataFrame, splits: Splits) -> None:
    indices = [set(part.index) for part in splits]
    assert sum(len(i) for i in indices) == len(cleaned)
    assert set().union(*indices) == set(cleaned.index)
    assert not indices[0] & indices[1]
    assert not indices[0] & indices[2]
    assert not indices[1] & indices[2]


def test_split_proportions_are_60_20_20(cleaned: pd.DataFrame, splits: Splits) -> None:
    n = len(cleaned)
    assert len(splits.train) == pytest.approx(0.6 * n, abs=1)
    assert len(splits.val) == pytest.approx(0.2 * n, abs=1)
    assert len(splits.test) == pytest.approx(0.2 * n, abs=1)


def test_split_preserves_churn_rate(cleaned: pd.DataFrame, splits: Splits) -> None:
    overall = cleaned[TARGET].mean()
    for part in splits:
        assert part[TARGET].mean() == pytest.approx(overall, abs=0.03)


def test_split_is_deterministic(cleaned: pd.DataFrame) -> None:
    first = split(cleaned, seed=7)
    second = split(cleaned, seed=7)
    for a, b in zip(first, second, strict=True):
        assert list(a.index) == list(b.index)


def test_different_seeds_give_different_splits(cleaned: pd.DataFrame) -> None:
    assert list(split(cleaned, seed=1).train.index) != list(split(cleaned, seed=2).train.index)

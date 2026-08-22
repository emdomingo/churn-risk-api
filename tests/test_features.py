"""Tests for the feature encoder.

The encoder is fitted once at training time and shipped inside the artifact, so the bugs
worth guarding against are the silent ones: a column that means something different at
inference than it did at training, or a record that cannot be scored at all.
"""

import pandas as pd
import pytest

from churn.data import TARGET, split
from churn.features import (
    NUMERIC_FEATURES,
    build_preprocessor,
    categorical_features,
    split_xy,
)


@pytest.fixture(scope="module")
def encoder_and_frame(sample_frame: pd.DataFrame):
    X, _ = split_xy(sample_frame)
    return build_preprocessor(sample_frame).fit(X), X


def test_categorical_features_excludes_numerics_and_target(sample_frame: pd.DataFrame) -> None:
    categorical = categorical_features(sample_frame)
    assert TARGET not in categorical
    assert not set(categorical) & set(NUMERIC_FEATURES)
    assert len(categorical) + len(NUMERIC_FEATURES) + 1 == len(sample_frame.columns)


def test_split_xy_separates_target(sample_frame: pd.DataFrame) -> None:
    X, y = split_xy(sample_frame)
    assert TARGET not in X.columns
    assert y.name == TARGET
    assert len(X) == len(y) == len(sample_frame)


def test_output_is_a_named_frame(encoder_and_frame) -> None:
    """Driver names come straight off these columns, so a bare array would lose them."""
    encoder, X = encoder_and_frame
    encoded = encoder.transform(X)
    assert isinstance(encoded, pd.DataFrame)
    assert list(encoded.columns) == list(encoder.get_feature_names_out())


def test_numeric_columns_pass_through_unchanged(encoder_and_frame) -> None:
    """Unscaled on purpose: `/explain` reports tenure in months, not standard deviations."""
    encoder, X = encoder_and_frame
    encoded = encoder.transform(X)
    for column in NUMERIC_FEATURES:
        pd.testing.assert_series_equal(
            encoded[column], X[column], check_names=False, check_dtype=False
        )


def test_feature_names_are_unprefixed_column_level_pairs(encoder_and_frame) -> None:
    encoder, _ = encoder_and_frame
    names = list(encoder.get_feature_names_out())
    assert not any(name.startswith(("numeric__", "categorical__")) for name in names)
    assert "Contract_Month-to-month" in names


def test_every_level_is_kept(encoder_and_frame, sample_frame: pd.DataFrame) -> None:
    """No reference level dropped — a level absent from the matrix could never be named
    as a driver."""
    encoder, _ = encoder_and_frame
    names = set(encoder.get_feature_names_out())
    for level in sample_frame["Contract"].unique():
        assert f"Contract_{level}" in names


def test_unknown_level_encodes_to_zeros_instead_of_raising(encoder_and_frame) -> None:
    """A novel category must not 500 the API. It degrades to 'none of the known levels'."""
    encoder, X = encoder_and_frame
    record = X.iloc[[0]].copy()
    record["Contract"] = "Five year moon lease"

    encoded = encoder.transform(record)

    contract_columns = [c for c in encoded.columns if c.startswith("Contract_")]
    assert encoded[contract_columns].to_numpy().sum() == 0
    assert encoded.shape == (1, len(encoder.get_feature_names_out()))


def test_column_order_is_stable_across_splits(sample_frame: pd.DataFrame) -> None:
    """Train and serve must agree on position, not just on names."""
    splits = split(sample_frame)
    X_train, _ = split_xy(splits.train)
    X_val, _ = split_xy(splits.val)
    encoder = build_preprocessor(sample_frame).fit(X_train)

    assert list(encoder.transform(X_train).columns) == list(encoder.transform(X_val).columns)


def test_transform_ignores_column_order_of_the_input(encoder_and_frame) -> None:
    """Records arrive as JSON objects, where key order is not meaningful."""
    encoder, X = encoder_and_frame
    shuffled = X.iloc[[0]][list(reversed(X.columns))]
    pd.testing.assert_frame_equal(encoder.transform(shuffled), encoder.transform(X.iloc[[0]]))

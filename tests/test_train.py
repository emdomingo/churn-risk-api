"""Tests for the training path and the artifact it produces.

Not tests of model quality — a booster fitted on 200 rows has none. These check the three
things that would be wrong-answer bugs in production rather than visible failures: that
the artifact survives a round-trip intact, that the persisted explainer explains the same
model the score came from, and that the threshold really is derived from the stated cost
assumption rather than hardcoded.
"""

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from churn.artifact import ChurnArtifact, load, save
from churn.features import split_xy
from churn.train import (
    MISSED_CHURNER_COST_RATIO,
    build_artifact,
    evaluate,
    format_report,
    operating_threshold,
)

# --- the operating threshold (gate #5) ---------------------------------------------


def test_threshold_matches_the_cost_ratio() -> None:
    """p > 1 / (1 + ratio). Equal costs is the familiar 0.5; 5x lands at 1/6."""
    assert operating_threshold(1.0) == pytest.approx(0.5)
    assert operating_threshold(5.0) == pytest.approx(1 / 6)
    assert operating_threshold() == pytest.approx(1 / (1 + MISSED_CHURNER_COST_RATIO))


def test_a_costlier_miss_lowers_the_threshold() -> None:
    """The direction is the whole point: the more a lost customer hurts, the more
    willing you are to spend offers on false alarms."""
    ratios = [1.0, 5.0, 20.0]
    thresholds = [operating_threshold(r) for r in ratios]
    assert thresholds == sorted(thresholds, reverse=True)


# --- metrics -----------------------------------------------------------------------


def test_evaluate_counts_a_known_confusion_matrix() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.10, 0.90, 0.90, 0.10])

    result = evaluate(y_true, probabilities, threshold=0.5)

    assert result["confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["flagged_share"] == pytest.approx(0.5)


def test_evaluate_survives_a_split_where_nothing_is_flagged() -> None:
    """Pinned labels: an unpinned confusion matrix returns 1x1 here and the unpack raises."""
    result = evaluate(np.array([0, 0, 1]), np.array([0.01, 0.02, 0.03]), threshold=0.5)
    assert result["confusion"] == {"tn": 2, "fp": 0, "fn": 1, "tp": 0}
    assert result["precision"] == 0.0


# --- the artifact ------------------------------------------------------------------


def test_artifact_carries_everything_needed_to_serve(tiny_artifact: ChurnArtifact) -> None:
    assert isinstance(tiny_artifact.calibrator, CalibratedClassifierCV)
    assert isinstance(tiny_artifact.booster, XGBClassifier)
    assert tiny_artifact.model_version == "test-artifact"
    assert tiny_artifact.trained_at.endswith("+00:00")
    assert tiny_artifact.feature_names == list(tiny_artifact.preprocessor.get_feature_names_out())


def test_metrics_block_has_what_the_readme_needs(tiny_artifact: ChurnArtifact) -> None:
    test_metrics = tiny_artifact.metrics["test"]
    assert {"roc_auc", "pr_auc", "brier", "confusion", "threshold"} <= set(test_metrics)
    assert tiny_artifact.metrics["calibration_method"] == "sigmoid"
    assert tiny_artifact.metrics["n_trees"] >= 1
    assert format_report(tiny_artifact).startswith("model_version   test-artifact")


def test_scores_are_probabilities(tiny_artifact: ChurnArtifact, sample_frame) -> None:
    X, _ = split_xy(sample_frame)
    scores = tiny_artifact.calibrator.predict_proba(tiny_artifact.transform(X))[:, 1]
    assert scores.shape == (len(X),)
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_calibration_actually_changes_the_scores(
    tiny_artifact: ChurnArtifact, sample_frame
) -> None:
    """If these matched, the calibrator would be a no-op and the Brier improvement a lie."""
    X, _ = split_xy(sample_frame)
    encoded = tiny_artifact.transform(X)
    calibrated = tiny_artifact.calibrator.predict_proba(encoded)[:, 1]
    raw = tiny_artifact.booster.predict_proba(encoded)[:, 1]
    assert not np.allclose(calibrated, raw)


def test_calibration_preserves_ranking(tiny_artifact: ChurnArtifact, sample_frame) -> None:
    """Sigmoid calibration is monotone. This is the licence for SHAP to explain the
    uncalibrated margin while the API returns a calibrated score."""
    X, _ = split_xy(sample_frame)
    encoded = tiny_artifact.transform(X)
    calibrated = tiny_artifact.calibrator.predict_proba(encoded)[:, 1]
    raw = tiny_artifact.booster.predict_proba(encoded)[:, 1]
    assert np.array_equal(np.argsort(calibrated, kind="stable"), np.argsort(raw, kind="stable"))


def test_explainer_explains_the_shipped_model(tiny_artifact: ChurnArtifact, sample_frame) -> None:
    """SHAP values plus the base value must reconstruct the booster's raw margin.

    This is the test that catches the early-stopping trap: if the model kept trees the
    prediction path ignores, the explainer would read them and these numbers would drift
    apart.
    """
    X, _ = split_xy(sample_frame)
    encoded = tiny_artifact.transform(X.head(20))

    explanation = tiny_artifact.explainer(encoded)
    reconstructed = explanation.values.sum(axis=1) + np.ravel(explanation.base_values)
    margin = tiny_artifact.booster.predict(encoded, output_margin=True)

    assert explanation.values.shape == (20, len(tiny_artifact.feature_names))
    assert np.allclose(reconstructed, margin, atol=1e-4)


def test_artifact_survives_a_round_trip(tiny_artifact: ChurnArtifact, sample_frame, tmp_path):
    """The service loads this file at import; a lossy round-trip would surface as wrong
    scores in production rather than as an error here."""
    X, _ = split_xy(sample_frame)
    path = save(tiny_artifact, tmp_path / "model.joblib")
    reloaded = load(path)

    assert reloaded.model_version == tiny_artifact.model_version
    assert reloaded.feature_names == tiny_artifact.feature_names
    np.testing.assert_array_equal(
        reloaded.calibrator.predict_proba(reloaded.transform(X))[:, 1],
        tiny_artifact.calibrator.predict_proba(tiny_artifact.transform(X))[:, 1],
    )


def test_load_explains_how_to_build_a_missing_artifact(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="churn.train"):
        load(tmp_path / "absent.joblib")


def test_training_is_deterministic(sample_frame) -> None:
    """Same seed, same data, same scores — reproducibility is part of the grade."""
    X, _ = split_xy(sample_frame)
    first = build_artifact(sample_frame, model_version="a")
    second = build_artifact(sample_frame, model_version="b")
    np.testing.assert_array_equal(
        first.calibrator.predict_proba(first.transform(X))[:, 1],
        second.calibrator.predict_proba(second.transform(X))[:, 1],
    )

"""The single model artifact: what gets persisted, and how it is stamped.

Everything needed to serve a request lives in one file — encoder, calibrated model,
explainer, feature names, and the metadata describing how they were produced. One file
means one load and, more importantly, no way for the pieces to drift out of sync: an
encoder that no longer matches the model it was fitted with is a silent wrong-answer bug,
not a crash.

The cost of `joblib` is that a pickle is tied to the library versions that wrote it. That
risk is contained here because the artifact is *built inside the same image that loads
it* — the Docker build trains from the same `uv.lock` that the runtime stage installs, so
writer and reader can never disagree. Nothing unpickles an artifact from elsewhere.
"""

import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "model.joblib"
RELIABILITY_PLOT_PATH = ARTIFACT_DIR / "reliability.png"


@dataclass(frozen=True)
class ChurnArtifact:
    """Everything the service needs at inference, loaded once at module import."""

    preprocessor: ColumnTransformer
    calibrator: CalibratedClassifierCV
    explainer: Any  # shap.TreeExplainer — typed loosely to keep shap out of this module
    feature_names: list[str]
    model_version: str
    trained_at: str
    seed: int
    metrics: dict[str, Any]

    @property
    def booster(self) -> Any:
        """The raw XGBoost model underneath the calibrator.

        Two layers to unwrap: `CalibratedClassifierCV` holds one fitted
        `_CalibratedClassifier`, whose `.estimator` is the `FrozenEstimator` shell around
        the booster. This is the object `TreeExplainer` runs on — which is exactly why
        the drivers explain the *uncalibrated* margin while the score is calibrated.
        """
        frozen = self.calibrator.calibrated_classifiers_[0].estimator
        return getattr(frozen, "estimator", frozen)

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        """Encode raw customer records into the model's feature space."""
        return self.preprocessor.transform(records)


def resolve_model_version() -> str:
    """Stamp identifying the code that produced this artifact.

    `MODEL_VERSION` from the environment wins — that is how the Docker build injects the
    CI commit, since `.git` never enters the image. Otherwise the package version is
    combined with the current short commit, which is the part that actually answers "did
    my deploy land?": a semver alone would only change when someone remembers to bump it,
    and PLAN's verification step requires the live `model_version` to change on every
    merge. A dirty working tree is marked, because an artifact trained from uncommitted
    code is not reproducible from the commit it names.
    """
    from os import environ

    override = environ.get("MODEL_VERSION")
    if override:
        return override

    try:
        base = package_version("churn")
    except PackageNotFoundError:  # pragma: no cover - only if run from an uninstalled tree
        base = "0.0.0"

    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return f"{base}+dev"

    return f"{base}+g{commit}.dirty" if dirty else f"{base}+g{commit}"


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def save(artifact: ChurnArtifact, path: Path = ARTIFACT_PATH) -> Path:
    """Write the artifact, creating `artifacts/` if needed. Never enters git."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load(path: Path = ARTIFACT_PATH) -> ChurnArtifact:
    """Read the artifact. Called once at import time, not per request."""
    if not path.exists():
        raise FileNotFoundError(
            f"No model artifact at {path}. Build one with: "
            "uv run --group train python -m churn.train"
        )
    return joblib.load(path)

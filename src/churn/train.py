"""Train, calibrate, explain, persist — the whole offline path in one command.

    uv run --group train python -m churn.train

Four decisions are baked in here, each with a plainer alternative that was considered and
rejected. They are the ones the README has to defend, so they are stated at the top
rather than buried in the code.

**Sigmoid calibration, not isotonic.** The calibrator is fitted on the validation split:
~1,400 rows, ~370 of them churners. Isotonic would fit an arbitrary monotone staircase
and generally scores a better Brier when it has the data — but at this sample size it
overfits, it collapses the output onto a few dozen distinct values (so customers inside a
step become unrankable, and a risk-band cut can land mid-plateau), and it can emit hard
0.0 and 1.0. Sigmoid fits two parameters. It is also the *right shape* for the
miscalibration actually present: `scale_pos_weight` inflates the model's opinion of
churners by roughly a constant shift in log-odds, which is precisely what a two-parameter
logistic undoes. The method cannot be chosen empirically here — val is where the
calibrator is fitted, and picking on test would be selection on the final holdout.

**Early stopping on a slice carved out of train — never on val.** A fixed tree budget was
tried first and measurably failed: 300 trees at depth 4 gave train ROC-AUC 0.924 against
test 0.818, and test AUC *fell* monotonically as trees were added (0.832 at 50 trees,
0.818 at 300). This dataset is small and its signal concentrated, so boosting overfits
early; "conservative fixed budget" turned out not to be conservative at all. The fix
holds the important line — `val` is still touched by nothing but the calibrator, so the
calibration story stays clean — and pays for it with 20% of the *training* rows, which
become an internal stopping slice. Depth 2 and the tree count are both chosen by that
slice, scored on PR-AUC: the booster's job is ranking under a 26.5% base rate, and
turning a ranking into a probability is the calibrator's job downstream.

Two passes, not one. Early stopping halts training `EARLY_STOPPING_ROUNDS` after the best
iteration and leaves those extra trees in the model — prediction ignores them, but
`TreeExplainer` reads the whole dump, so the drivers would explain a model the score never
came from. So the stopping slice only *chooses the number*; the shipped model is refit on
all of train with exactly that many trees, and every tree in it is used.

**The operating threshold comes from a cost ratio,** not from 0.5 and not from maximising
F1. See `MISSED_CHURNER_COST_RATIO` and `operating_threshold`.

**SHAP explains the uncalibrated margin.** `TreeExplainer` runs on the booster inside the
calibrator, so the returned score is calibrated while the drivers explain the
pre-calibration margin. Calibration is monotone, so driver *ranking* is unaffected — but
the contribution magnitudes are in margin units, not probability units. This must be
stated in the README, never glossed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from churn.artifact import (
    RELIABILITY_PLOT_PATH,
    ChurnArtifact,
    resolve_model_version,
    save,
    utc_timestamp,
)
from churn.data import RANDOM_SEED, TARGET, Splits, clean, load_raw, split
from churn.features import build_preprocessor, split_xy

# See the module docstring. The tree count is chosen by the stopping slice, so MAX_TREES
# is only a ceiling. Depth 2 was selected by the same slice from {2, 3, 4} — at 46 mostly
# one-hot columns over ~4k rows, deeper trees buy memorisation, not signal.
MAX_TREES = 1000
LEARNING_RATE = 0.05
MAX_DEPTH = 2
EARLY_STOPPING_ROUNDS = 50
STOP_SLICE_FRACTION = 0.2
STOP_METRIC = "aucpr"

CALIBRATION_METHOD = "sigmoid"

# Gate #5, the confusion-matrix threshold. This is a *business assumption*, written down
# as one number so it can be argued with: missing a churner is assumed 5x as costly as
# spending a retention offer on someone who was never going to leave. Roughly, that is
#   (customer lifetime value x probability the offer actually works) / cost of the offer,
# and the middle term is the one being waved at — a strict treatment would model offer
# acceptance separately. 5 is deliberately conservative for telco, where CLV over offer
# cost alone is nearer 10-20.
MISSED_CHURNER_COST_RATIO = 5.0

TOP_DRIVER_COUNT = 10


def operating_threshold(cost_ratio: float = MISSED_CHURNER_COST_RATIO) -> float:
    """Cost-minimising cutoff for a *calibrated* score — a closed form, not a search.

    Flagging costs the price of one offer with probability `1 - p` (the customer was
    staying anyway); not flagging costs `cost_ratio` offers with probability `p`. Flag
    when `cost_ratio * p > 1 - p`, i.e. when `p > 1 / (1 + cost_ratio)`.

    This is the payoff of calibration, and the reason to prefer it over maximising F1:
    the threshold falls out of a stated business assumption rather than being tuned on
    the holdout, and F1's implicit assumption — that a missed churner and a wasted offer
    cost the same — is one nobody in retention would actually agree to.
    """
    return 1.0 / (1.0 + cost_ratio)


def _new_booster(n_estimators: int, scale_pos_weight: float, seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric=STOP_METRIC,
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
    )


def choose_tree_count(X: pd.DataFrame, y: pd.Series, seed: int = RANDOM_SEED) -> int:
    """How many trees, decided by a stopping slice taken out of the *training* split.

    Nothing here ever sees `val` or `test`. The returned count is the last iteration that
    improved PR-AUC on the slice; the model that used it is then thrown away.
    """
    fit_rows, stop_rows = train_test_split(
        np.arange(len(X)),
        test_size=STOP_SLICE_FRACTION,
        stratify=y,
        random_state=seed,
    )
    y_fit = y.iloc[fit_rows]
    probe = _new_booster(
        MAX_TREES,
        scale_pos_weight=float((y_fit == 0).sum() / (y_fit == 1).sum()),
        seed=seed,
    )
    probe.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    probe.fit(
        X.iloc[fit_rows],
        y_fit,
        eval_set=[(X.iloc[stop_rows], y.iloc[stop_rows])],
        verbose=False,
    )
    return int(probe.best_iteration) + 1


def fit_booster(
    X: pd.DataFrame, y: pd.Series, n_estimators: int, seed: int = RANDOM_SEED
) -> XGBClassifier:
    """Fit the shipped booster on the whole training split, with a fixed tree count.

    `scale_pos_weight` is derived from this split's own class counts rather than the
    hardcoded 2.768 from the exploration notebook, so the value stays correct when the
    seed, the split fractions, or the data change.

    The count comes from `choose_tree_count`, which saw only 80% of these rows — applying
    it to 100% is mildly conservative, and the alternative (discarding the stopping slice
    from the final fit) throws away ~845 real training rows to avoid a rounding error.
    """
    negatives, positives = int((y == 0).sum()), int((y == 1).sum())
    booster = _new_booster(n_estimators, scale_pos_weight=negatives / positives, seed=seed)
    booster.fit(X, y)
    return booster


def calibrate(booster: XGBClassifier, X: pd.DataFrame, y: pd.Series) -> CalibratedClassifierCV:
    """Fit the probability calibrator on the validation split.

    `FrozenEstimator` is what keeps this a *single* model: it tells sklearn the booster is
    already fitted, so only the two sigmoid parameters are learned here and the booster
    itself is untouched. (`cv="prefit"` did this job until sklearn 1.9 removed it.) The
    rejected alternative, `cv=5`, would cross-fit five boosters and average them — an
    ensemble by another name, and `TreeExplainer` explains one booster, not five.
    """
    return CalibratedClassifierCV(FrozenEstimator(booster), method=CALIBRATION_METHOD).fit(X, y)


def evaluate(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    """Metrics for the calibrated scores on a held-out split.

    ROC-AUC, PR-AUC and Brier answer three different questions: ranking quality, ranking
    quality where the positives are rare, and whether the numbers are probabilities at
    all. Accuracy answers none of them — at a 26.5% base rate, predicting "nobody churns"
    scores 73.5%.
    """
    # labels pinned: on a split small enough that one class is never predicted, an
    # unpinned confusion matrix comes back 1x1 and the unpack raises.
    tn, fp, fn, tp = confusion_matrix(y_true, probabilities >= threshold, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "base_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "threshold": float(threshold),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "flagged_share": float((tp + fp) / len(y_true)),
    }


def tenure_only_roc_auc(splits: Splits) -> float:
    """The floor from the exploration notebook, recomputed on the *test* split.

    The notebook's 0.740 was measured over all 7,043 rows while the model's AUC is a
    test-split number, so comparing them directly would be apples to oranges. Tenure runs
    the other way (longer tenure, less churn), hence the sign flip.
    """
    return float(roc_auc_score(splits.test[TARGET], -splits.test["tenure"]))


def global_drivers(booster: XGBClassifier, encoded: pd.DataFrame) -> list[tuple[str, float]]:
    """Mean |SHAP| per feature on the training split — the ranking to read against §8 of
    `exploration.ipynb`. Global importance only; per-request drivers are S3's job."""
    values = shap.TreeExplainer(booster).shap_values(encoded)
    ranked = pd.Series(np.abs(values).mean(axis=0), index=encoded.columns).sort_values(
        ascending=False
    )
    return [(name, float(value)) for name, value in ranked.head(TOP_DRIVER_COUNT).items()]


def build_artifact(
    df: pd.DataFrame, seed: int = RANDOM_SEED, model_version: str | None = None
) -> ChurnArtifact:
    """Full offline path as one pure function: cleaned frame in, artifact out.

    No file IO and no printing, so the test suite can build a small artifact from the
    200-row fixture through the identical code path that produces the real one.
    """
    splits = split(df, seed=seed)
    X_train, y_train = split_xy(splits.train)
    X_val, y_val = split_xy(splits.val)
    X_test, y_test = split_xy(splits.test)

    preprocessor = build_preprocessor(df).fit(X_train)
    encoded_train = preprocessor.transform(X_train)
    encoded_val = preprocessor.transform(X_val)
    encoded_test = preprocessor.transform(X_test)

    n_trees = choose_tree_count(encoded_train, y_train, seed=seed)
    booster = fit_booster(encoded_train, y_train, n_estimators=n_trees, seed=seed)
    calibrator = calibrate(booster, encoded_val, y_val)

    threshold = operating_threshold()
    calibrated = calibrator.predict_proba(encoded_test)[:, 1]
    uncalibrated = booster.predict_proba(encoded_test)[:, 1]

    metrics = {
        "test": evaluate(y_test, calibrated, threshold),
        "uncalibrated": {
            "roc_auc": float(roc_auc_score(y_test, uncalibrated)),
            "brier": float(brier_score_loss(y_test, uncalibrated)),
        },
        "train_roc_auc": float(
            roc_auc_score(y_train, booster.predict_proba(encoded_train)[:, 1])
        ),
        "baseline_tenure_only_roc_auc": tenure_only_roc_auc(splits),
        "scale_pos_weight": float(booster.get_params()["scale_pos_weight"]),
        "cost_ratio": MISSED_CHURNER_COST_RATIO,
        "calibration_method": CALIBRATION_METHOD,
        "n_trees": n_trees,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "split_sizes": {
            "train": len(splits.train),
            "val": len(splits.val),
            "test": len(splits.test),
        },
        "top_drivers": global_drivers(booster, encoded_train),
    }

    return ChurnArtifact(
        preprocessor=preprocessor,
        calibrator=calibrator,
        explainer=shap.TreeExplainer(booster),
        feature_names=list(encoded_train.columns),
        model_version=model_version or resolve_model_version(),
        trained_at=utc_timestamp(),
        seed=seed,
        metrics=metrics,
    )


def format_report(artifact: ChurnArtifact) -> str:
    """The metrics block, shaped to paste straight into the README."""
    m = artifact.metrics
    test, confusion = m["test"], m["test"]["confusion"]
    sizes = m["split_sizes"]

    lines = [
        f"model_version   {artifact.model_version}",
        f"trained_at      {artifact.trained_at}",
        f"splits          train {sizes['train']} / val {sizes['val']} / test {sizes['test']}"
        f"   seed {artifact.seed}",
        f"imbalance       scale_pos_weight {m['scale_pos_weight']:.3f}"
        f"   base rate {test['base_rate']:.4f}",
        f"booster         {m['n_trees']} trees, depth {m['max_depth']}, lr {m['learning_rate']}"
        f"   (count chosen on a {STOP_SLICE_FRACTION:.0%} slice of train, {STOP_METRIC})",
        f"calibration     {m['calibration_method']} on val, fitted over a frozen booster",
        "",
        "| Metric | Test | Reads as |",
        "|---|---|---|",
        f"| ROC-AUC | {test['roc_auc']:.4f} | ranking quality over all thresholds |",
        f"| PR-AUC | {test['pr_auc']:.4f} | ranking quality on the ~26% that matter"
        f" (no-skill = {test['base_rate']:.4f}) |",
        f"| Brier | {test['brier']:.4f} | are the scores probabilities"
        f" (uncalibrated {m['uncalibrated']['brier']:.4f}) |",
        f"| ROC-AUC, uncalibrated | {m['uncalibrated']['roc_auc']:.4f} |"
        " calibration is monotone, so ranking is unchanged |",
        f"| tenure-only baseline | {m['baseline_tenure_only_roc_auc']:.4f} |"
        " the floor from exploration.ipynb, recomputed on test |",
        f"| ROC-AUC, train split | {m['train_roc_auc']:.4f} |"
        " overfit check: the gap to test is the memorisation |",
        "",
        f"Confusion matrix at p >= {test['threshold']:.4f}"
        f"  (1 / (1 + {m['cost_ratio']:.0f}); a missed churner is assumed"
        f" {m['cost_ratio']:.0f}x the cost of a wasted offer)",
        "",
        "|  | predicted stay | predicted churn |",
        "|---|---|---|",
        f"| **actually stayed** | {confusion['tn']} | {confusion['fp']} |",
        f"| **actually churned** | {confusion['fn']} | {confusion['tp']} |",
        "",
        f"recall {test['recall']:.3f}   precision {test['precision']:.3f}"
        f"   flagged {test['flagged_share']:.1%} of the base",
        "",
        f"Top {TOP_DRIVER_COUNT} global drivers (mean |SHAP| on train, margin units)",
    ]
    width = max(len(name) for name, _ in m["top_drivers"])
    lines += [
        f"  {rank:>2}. {name:<{width}}  {value:.4f}"
        for rank, (name, value) in enumerate(m["top_drivers"], start=1)
    ]
    return "\n".join(lines)


def write_reliability_diagram(
    artifact: ChurnArtifact, df: pd.DataFrame, path: Path = RELIABILITY_PLOT_PATH
) -> Path:
    """Calibrated vs. uncalibrated on the test split, against the diagonal.

    matplotlib is imported lazily: it lives in the `train` dependency group and is
    deliberately absent from the Lambda image, so this module must stay importable
    without it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
    calibrated_colour, raw_colour = "#2a78d6", "#eb6834"

    splits = split(df, seed=artifact.seed)
    X_test, y_test = split_xy(splits.test)
    encoded = artifact.transform(X_test)
    calibrated = artifact.calibrator.predict_proba(encoded)[:, 1]
    uncalibrated = artifact.booster.predict_proba(encoded)[:, 1]

    fig, ax = plt.subplots(figsize=(5.6, 5.2), facecolor=surface)
    ax.set_facecolor(surface)
    ax.plot([0, 1], [0, 1], color=muted, linewidth=1, linestyle="--", label="perfect")
    for probabilities, colour, label in (
        (uncalibrated, raw_colour, "uncalibrated booster"),
        (calibrated, calibrated_colour, f"{CALIBRATION_METHOD} calibrated"),
    ):
        observed, predicted = calibration_curve(
            y_test, probabilities, n_bins=10, strategy="quantile"
        )
        ax.plot(predicted, observed, marker="o", markersize=4, color=colour, label=label)

    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed churn rate")
    ax.set_title("Reliability on the test split", loc="left", color=ink, fontweight="semibold")
    ax.grid(color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
    return path


def main() -> None:
    df = clean(load_raw())
    artifact = build_artifact(df)
    artifact_path = save(artifact)
    plot_path = write_reliability_diagram(artifact, df)

    print(format_report(artifact))
    print()
    print(f"wrote {artifact_path}  ({artifact_path.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()

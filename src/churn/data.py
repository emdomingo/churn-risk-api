"""Load, clean, and split the Telco customer churn dataset.

Kept deliberately free of model and framework code: `clean` is the only place the
dataset's quirks are handled, so tests can exercise it against a hand-built frame and
the fixture path and production path run identical logic.
"""

from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "Telco-Customer-Churn.csv"

TARGET = "Churn"
ID_COLUMN = "customerID"
RANDOM_SEED = 42

# 60/20/20. The validation split is structurally required, not a nicety: probability
# calibration must be fitted on data the booster never saw.
VAL_FRACTION = 0.2
TEST_FRACTION = 0.2


class Splits(NamedTuple):
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Read the CSV with no transformation applied."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a typed, model-ready frame. Does not mutate the input.

    Three dataset quirks are handled here:

    1. `TotalCharges` is string-typed with 11 blank values. Every blank row has
       `tenure == 0`, so the value is not missing at random — the customer has been
       billed for zero months. Filled with 0.0, which is the semantically correct
       amount rather than an estimate. This also keeps `tenure == 0` customers
       scoreable at inference, where they are a real and interesting retention target.
    2. `SeniorCitizen` ships as int 0/1 while every other binary column is Yes/No.
       Normalised here so the feature pipeline sees one consistent shape.
    3. `customerID` is an identifier with no predictive content.
    """
    out = df.copy()

    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)
    out["TotalCharges"] = out["TotalCharges"].astype("float64")

    if "SeniorCitizen" in out.columns:
        out["SeniorCitizen"] = out["SeniorCitizen"].map({0: "No", 1: "Yes"}).astype("object")

    if ID_COLUMN in out.columns:
        out = out.drop(columns=[ID_COLUMN])

    out[TARGET] = out[TARGET].map({"No": 0, "Yes": 1}).astype("int64")

    return out


def split(df: pd.DataFrame, seed: int = RANDOM_SEED) -> Splits:
    """Stratified 60/20/20 split on the target, deterministic for a given seed."""
    holdout_fraction = VAL_FRACTION + TEST_FRACTION
    train, holdout = train_test_split(
        df,
        test_size=holdout_fraction,
        stratify=df[TARGET],
        random_state=seed,
    )
    val, test = train_test_split(
        holdout,
        test_size=TEST_FRACTION / holdout_fraction,
        stratify=holdout[TARGET],
        random_state=seed,
    )
    return Splits(train=train, val=val, test=test)

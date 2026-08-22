"""Feature encoding for the churn model.

A single `ColumnTransformer`, fitted on the training split and then persisted inside the
model artifact. Fitting it once and shipping it is what guarantees the encoding at
inference is byte-identical to the encoding at training time — the classic serving skew
bug has no room to appear.

Three encoding choices matter downstream, and all three are made for the sake of
`/explain` rather than for accuracy:

1. **Numerics pass through unscaled.** Trees split on order, so scaling is a no-op for
   the model — but it is not a no-op for the explanation. A driver reading
   `tenure = 2 months` is actionable; `tenure = -1.31` (standardised) is not.
2. **Every categorical level is kept** (`drop=None`). Dropping a reference level would
   remove it from the feature matrix entirely, so SHAP could never name it — a driver
   list that is structurally incapable of saying `Contract=Two year` is a worse
   explanation. The resulting collinearity is harmless to a tree, and 46 columns from
   19 is nothing at this size.
3. **Unknown levels encode to all-zeros** (`handle_unknown="ignore"`) rather than
   raising. A customer record arriving with a level the training data never saw must
   still be scoreable — the API cannot 500 on novel input. The cost is that the record
   is scored as if that column were "none of the known levels", which is a quiet
   degradation rather than a loud failure. Deliberate: a slightly-off score beats a 500.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from churn.data import TARGET

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]


def categorical_features(df: pd.DataFrame) -> list[str]:
    """Every column that is neither numeric nor the target, in frame order."""
    excluded = {*NUMERIC_FEATURES, TARGET}
    return [column for column in df.columns if column not in excluded]


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature frame from the target column."""
    return df.drop(columns=[TARGET]), df[TARGET]


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Return an unfitted encoder for the columns of `df`.

    Outputs a named DataFrame, not a bare array: the column names *are* the driver names
    `/explain` reports, and `verbose_feature_names_out=False` keeps them as
    `Contract_Month-to-month` rather than `cat__Contract_Month-to-month`.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None),
                categorical_features(df),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

"""Regenerate the ~200-row test fixture from the full dataset.

Run from the repo root: `uv run python scripts/make_fixture.py`

The sampling is not uniform, and deliberately so. All 11 blank-`TotalCharges` rows are
force-included, because a uniform 200-row draw from 7043 would be expected to contain
0.31 of them — the blank-handling test would almost certainly pass vacuously. The
remainder is stratified on `Churn` to preserve the ~26.5% positive rate.

The fixture is written in RAW form (blank strings intact, `customerID` present): it is
the input to `churn.data.clean`, not its output.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.data import RAW_PATH, TARGET

OUT_PATH = RAW_PATH.parents[1] / "tests" / "fixtures" / "telco_sample.csv"
FIXTURE_ROWS = 200
SEED = 42


def main() -> None:
    df = pd.read_csv(RAW_PATH)

    blank = pd.to_numeric(df["TotalCharges"], errors="coerce").isna()
    forced = df[blank]
    remainder = df[~blank]

    sampled, _ = train_test_split(
        remainder,
        train_size=FIXTURE_ROWS - len(forced),
        stratify=remainder[TARGET],
        random_state=SEED,
    )

    fixture = pd.concat([forced, sampled]).sample(frac=1, random_state=SEED)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fixture.to_csv(OUT_PATH, index=False)

    print(f"wrote {len(fixture)} rows to {OUT_PATH}")
    print(f"  forced blank-TotalCharges rows: {len(forced)}")
    print(
        f"  churn rate: {fixture[TARGET].eq('Yes').mean():.4f} "
        f"(full: {df[TARGET].eq('Yes').mean():.4f})"
    )


if __name__ == "__main__":
    main()

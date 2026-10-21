"""Submission construction and validation.

The competition expects a ``submission.parquet`` file with exactly the
columns ``id`` and ``model`` — one row per test id, no nulls, and only the
18 valid model names.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ID_COL, LABELS, TARGET_COL


def make_submission(test: pd.DataFrame, predictions) -> pd.DataFrame:
    """Build the submission frame from test ids + predictions."""
    preds = pd.Series(np.asarray(predictions).ravel(), index=test.index)
    return pd.DataFrame({ID_COL: test[ID_COL].values, TARGET_COL: preds.values})


def validate_submission(sub: pd.DataFrame, expected_ids) -> list[str]:
    """Return a list of problems (empty list == valid).

    Checks columns, row count, id uniqueness/coverage, nulls and label validity.
    """
    problems: list[str] = []
    expected = set(expected_ids)

    if list(sub.columns) != [ID_COL, TARGET_COL]:
        problems.append(f"columns must be ['{ID_COL}', '{TARGET_COL}'], got {list(sub.columns)}")

    if sub[TARGET_COL].isna().any():
        problems.append("submission contains null model values")

    dupes = sub[ID_COL].duplicated().sum()
    if dupes:
        problems.append(f"{dupes} duplicated id(s)")

    got, missing = set(sub[ID_COL]), expected - set(sub[ID_COL])
    if missing:
        problems.append(f"{len(missing)} test id(s) missing from submission")
    extra = got - expected
    if extra:
        problems.append(f"{len(extra)} unknown id(s) in submission")

    bad = set(sub[TARGET_COL].dropna()) - set(LABELS)
    if bad:
        problems.append(f"{len(bad)} invalid label(s), e.g. {sorted(bad)[:3]}")

    return problems


def write_submission(sub: pd.DataFrame, path: str | Path = "submission.parquet") -> Path:
    """Validate semantics then write parquet (Kaggle code-competition default)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(path, index=False)
    return path

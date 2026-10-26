"""Experiment tracker — the instrument that compares local CV with the LB.

The competition gives five submissions a day and no labels.  The only way to
learn which facet combination works is to record, for each variant, both the
local prompt-grouped CV score and the public leaderboard score, and to watch
the *gap*.  This module is deliberately tiny: a CSV with one row per run, and
a report that keeps the latest row per variant.

A variant's configuration is hashed, so a notebook that was edited (views or
facets changed) cannot silently inherit an old score.

Typical use in a Kaggle notebook::

    from ntl import tracker

    row = tracker.make_row(
        variant="v5",
        views=("char", "word", "funcon"),
        facets=("length", "lexical", "punctuation", "structure", "emoji",
                "person", "hedging", "citation", "opener", "group"),
        local_cv=0.4897,
        local_folds=3,
        n_train=2699,
        n_pool=0,
    )
    print("TRACKER", tracker.to_json(row))
    tracker.append_csv("/kaggle/working/tracker.csv", row)

Then locally::

    uv run python scripts/track.py merge ~/Downloads/tracker.csv
    uv run python scripts/track.py report
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

TRACKER_COLUMNS = [
    "variant",
    "config_hash",
    "views",
    "facets",
    "facet_weight",
    "local_cv",
    "local_folds",
    "public_lb",
    "n_train",
    "n_pool",
    "run_at",
    "notes",
]

DEFAULT_PATH = Path("metrics/tracker.csv")


def config_hash(
    views: tuple[str, ...] | list[str],
    facets: tuple[str, ...] | list[str] = (),
    facet_weight: float = 1.0,
) -> str:
    """Short, stable hash of the parts of a variant that change its output."""
    payload = json.dumps(
        {
            "views": list(views),
            "facets": list(facets),
            "facet_weight": float(facet_weight),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def make_row(
    variant: str,
    views: tuple[str, ...] | list[str],
    facets: tuple[str, ...] | list[str] = (),
    facet_weight: float = 1.0,
    local_cv: float | None = None,
    local_folds: int | None = None,
    public_lb: float | None = None,
    n_train: int = 0,
    n_pool: int = 0,
    notes: str = "",
    run_at: str | None = None,
) -> dict:
    """Build one tracker row (all values JSON/CSV friendly)."""
    return {
        "variant": str(variant),
        "config_hash": config_hash(views, facets, facet_weight),
        "views": "+".join(views),
        "facets": "+".join(facets) if facets else "-",
        "facet_weight": float(facet_weight),
        "local_cv": None if local_cv is None else round(float(local_cv), 6),
        "local_folds": local_folds,
        "public_lb": None if public_lb is None else round(float(public_lb), 6),
        "n_train": int(n_train),
        "n_pool": int(n_pool),
        "run_at": run_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes": notes,
    }


def to_json(row: dict) -> str:
    """One-line JSON, for copying out of a Kaggle notebook log."""
    return json.dumps(row, sort_keys=True)


def append_csv(path: str | Path, row: dict) -> Path:
    """Append *row* to the tracker CSV, creating it with a header if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row]).reindex(columns=TRACKER_COLUMNS)
    header = not path.exists() or path.stat().st_size == 0
    frame.to_csv(path, mode="a", header=header, index=False)
    return path


def load(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Load the tracker, returning an empty typed frame if it does not exist."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=TRACKER_COLUMNS)
    frame = pd.read_csv(path)
    for column in TRACKER_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[TRACKER_COLUMNS]


def report(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Latest run per variant, with the local→public gap.

    ``gap`` is ``public_lb - local_cv``.  A consistently positive gap means
    the pool (or the hidden test being easier than train prompts) is helping;
    a negative gap means the local CV is optimistic.
    """
    frame = load(path)
    if frame.empty:
        return frame.assign(gap=pd.Series(dtype=float))
    frame = frame.sort_values("run_at", kind="stable")
    latest = frame.groupby("variant", as_index=False).tail(1).copy()
    latest["gap"] = latest["public_lb"] - latest["local_cv"]
    return latest.sort_values(
        ["public_lb", "local_cv"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)


def merge(path: str | Path, other: str | Path) -> int:
    """Append rows from another tracker CSV, skipping exact duplicates.

    Returns the number of rows added.  Used to pull a Kaggle-downloaded
    ``tracker.csv`` into the local record.
    """
    target = Path(path)
    incoming = load(other)
    if incoming.empty:
        return 0
    existing = load(target)
    keys = ["variant", "config_hash", "run_at", "local_cv", "public_lb"]
    known = set(map(tuple, existing[keys].astype(str).itertuples(index=False, name=None)))
    candidates = incoming[
        ~pd.Series(
            list(map(tuple, incoming[keys].astype(str).itertuples(index=False, name=None)))
        ).isin(known)
    ]
    if candidates.empty:
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    header = not target.exists() or target.stat().st_size == 0
    candidates.to_csv(target, mode="a", header=header, index=False)
    return int(len(candidates))

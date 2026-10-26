#!/usr/bin/env python
"""Generate the five submission variants: ``notebooks/04_v*.ipynb``.

Each variant is a **self-contained, uploadable notebook**: same plumbing, one
different hypothesis about which facet families carry author signal.  Upload
all five (or as many as the daily quota allows), read the public leaderboard,
and record the pair (local grouped CV, public LB) with ``scripts/track.py``.

The variants are chosen from measurements, not taste — every one of them has a
number attached in :mod:`ntl.measure` and in ``docs/HYPOTHESIS.md`` §9:

===========  ==========================================  ==================
variant      block added to the control stack             measured effect
===========  ==========================================  ==================
``v1``       nothing (control)                            reference 0.4464
``v2``       punctuation facets                           −0.014 (neutral)
``v3``       voice + punctuation                          +0.027
``v4``       voice + group deviation                      +0.026
``v5``       every response facet + group deviation       +0.043
===========  ==========================================  ==================

The individual blocks are *not* additive. Voice alone is 0.4386 and
punctuation alone is 0.4327 — both neutral against the 0.4464 control — but
together they reach 0.4730. The interesting hypothesis is an interaction, not
a main effect, and the five variants are chosen to expose it.

Usage:
    uv run python scripts/make_variants.py
    # then sync the inline package into each one:
    uv run python scripts/sync_notebook.py --all
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

ALL_FACETS = (
    "length",
    "lexical",
    "punctuation",
    "structure",
    "emoji",
    "person",
    "hedging",
    "citation",
    "opener",
    "group",
)

VOICE = ("length", "lexical", "person", "hedging", "opener")

VARIANTS: list[dict] = [
    {
        "id": "v1",
        "slug": "v1_control",
        "title": "v1 — control: the view stack",
        "views": ("char", "word", "funcon"),
        "facets": (),
        "hypothesis": "Reference point. The three n-gram views with prompt centring "
        "and within-group assignment, no numeric block at all.",
        "expected": "0.4464 at 3 folds (the 5-fold reference is 0.467).",
        "notes": "control",
    },
    {
        "id": "v2",
        "slug": "v2_punctuation",
        "title": "v2 — punctuation: the cheapest fingerprint, alone",
        "views": ("char", "word", "funcon"),
        "facets": ("punctuation",),
        "hypothesis": "Punctuation survives prompt control better than any other family "
        "(within-group η² 0.40 for the em dash) and scores 0.245 on its own. "
        "If any single family is a real author habit, it should be this one.",
        "expected": "0.4327 at 3 folds — measured neutral versus the 0.4464 control. "
        "The char view already encodes punctuation, so this is a redundancy test.",
        "notes": "punctuation only",
    },
    {
        "id": "v3",
        "slug": "v3_voice_punctuation",
        "title": "v3 — voice + punctuation: the first interaction",
        "views": ("char", "word", "funcon"),
        "facets": ("length", "lexical", "punctuation", "person", "hedging", "opener"),
        "hypothesis": "Neither block works alone: voice alone is 0.4386 and punctuation "
        "alone 0.4327, both neutral. The bet is that they are complementary — "
        "punctuation gives the classifier a clean cue, voice facets give it the "
        "register that makes the cue usable.",
        "expected": "0.4730 at 3 folds (+0.027 over control).",
        "notes": "voice + punctuation",
    },
    {
        "id": "v4",
        "slug": "v4_voice_deviation",
        "title": "v4 — voice + deviation: the relational interaction",
        "views": ("char", "word", "funcon"),
        "facets": ("length", "lexical", "person", "hedging", "opener", "group"),
        "hypothesis": "Absolute voice facets and group-relative deviation are neutral "
        "on their own (0.4386 and 0.4353) but answer different questions: how a "
        "model writes, and how it differs from its siblings on this prompt. "
        "Together they should reach the same place as voice + punctuation by a "
        "different route.",
        "expected": "0.4720 at 3 folds (+0.026 over control).",
        "notes": "voice + group deviation",
    },
    {
        "id": "v5",
        "slug": "v5_full",
        "title": "v5 — full: every facet, response and relational",
        "views": ("char", "word", "funcon"),
        "facets": ALL_FACETS,
        "hypothesis": "The complete hypothesised signal set: all ten response "
        "families (structure, emoji and citation are included even though they "
        "measure as null on this dataset) plus the group-relative block. If the "
        "interactions are real, this should beat every pair.",
        "expected": "0.4897 at 3 folds (+0.043 over control).",
        "notes": "all facets",
    },
]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


HEADER = """
# __TITLE__

**Variant `__ID__`** of the five-submission study described in
`docs/HYPOTHESIS.md` §9. This notebook is standalone: the `ntl` package is
inlined in the first cell, so it runs on Kaggle with no install and no
internet.

## Hypothesis

__HYPOTHESIS__

## Expected

__EXPECTED__

## What this notebook does

1. loads the competition train set and, if attached, the public in-style pool;
2. scores this exact configuration with **prompt-grouped** cross-validation
   (never random CV — that measures a different, much harder task);
3. refits on everything and writes `/kaggle/working/submission.parquet`;
4. prints a one-line `TRACKER_JSON` record and appends it to
   `/kaggle/working/tracker.csv`.

Run all five, then compare local CV against the public leaderboard. The
interesting quantity is the **gap**, not any single score.
"""

SETUP = """
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ntl import data, pipeline, submit, tracker
from ntl.config import LABELS

ON_KAGGLE = Path("/kaggle/input").exists()
WORKING = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")
WORKING.mkdir(parents=True, exist_ok=True)

print("data locations:")
for name, paths in data.data_locations().items():
    print(f"  {name}: {paths or '(not found)'}")
"""

CONFIG = """
VARIANT = "__ID__"
HYPOTHESIS = "__HYPOTHESIS__"

VIEWS = __VIEWS__
FACETS = __FACETS__
FACET_WEIGHT = 1.0

# Local CV is train-only and therefore comparable across variants.  Three
# folds keeps each notebook inside a few minutes; the ranking is stable.
RUN_LOCAL_CV = True
LOCAL_CV_FOLDS = 3

CONFIG_HASH = tracker.config_hash(VIEWS, FACETS, FACET_WEIGHT)
print(f"{VARIANT}  hash={CONFIG_HASH}")
print(f"views  : {VIEWS}")
print(f"facets : {FACETS or '(none)'}")
print(f"weight : {FACET_WEIGHT}")
print()
print(HYPOTHESIS)
"""

LOAD = """
train = data.load_train()
train_texts = data.as_text_array(train["text"])
train_labels = data.as_text_array(train["model"])
train_groups = data.as_text_array(train["prompt_id"])

raw_pool = data.load_in_style_pool()
print("raw pool:", data.describe_in_style_pool(raw_pool))

# Filter to the 18 labels, drop train duplicates, namespace pool prompt ids.
pool = data.prepare_training_pool(raw_pool, train_texts=train_texts)
print(f"usable pool: {len(pool)} rows")

if len(pool):
    texts = np.concatenate([train_texts, data.as_text_array(pool["text"])])
    labels = np.concatenate([train_labels, data.as_text_array(pool["model"])])
    group_ids = np.concatenate([train_groups, data.as_text_array(pool["prompt_id"])])
else:
    texts, labels, group_ids = train_texts, train_labels, train_groups

print(f"training on {len(texts)} passages ({len(train)} train + {len(pool)} pool)")
uniq, counts = np.unique(labels, return_counts=True)
print("labels:", dict(zip(uniq.tolist(), counts.tolist(), strict=True)))
"""

LOCAL_CV = """
if RUN_LOCAL_CV:
    cv_config = pipeline.StackConfig(
        views=VIEWS, facets=FACETS, facet_weight=FACET_WEIGHT
    )
    cv = pipeline.group_cv_score(
        train_texts, train_labels, train_groups, cv_config, n_splits=LOCAL_CV_FOLDS
    )
    local_cv = float(cv["balanced_accuracy"])
    local_folds = LOCAL_CV_FOLDS
    print(f"local grouped CV ({LOCAL_CV_FOLDS} folds): {local_cv:.4f}")
    print("folds:", [round(score, 4) for score in cv["fold_scores"]])
    print(f"chance: {1 / len(LABELS):.4f}")
else:
    local_cv, local_folds = None, None
    print("local CV skipped")
"""

SUBMIT = """
config = pipeline.StackConfig(views=VIEWS, facets=FACETS, facet_weight=FACET_WEIGHT)
model = pipeline.StackModel(config)
model.fit(texts, labels, groups=group_ids)

test = data.load_test()
test_texts = data.as_text_array(test["text"])
predictions = model.predict(test_texts)
print(f"predicted {len(predictions)} passages ({predictions.dtype})")

submission = submit.make_submission(test, predictions)
problems = submit.validate_submission(submission, test["id"])
print("validation problems:", problems or "none")

if ON_KAGGLE:
    out = WORKING / "submission.parquet"
else:
    out = WORKING / f"submission_{VARIANT}.parquet"
    print("NOTE: the local test.csv is a 3-row stub — this file is not a real submission.")

submit.write_submission(submission, out)
print("written:", out)
print()
print(submission["model"].value_counts().to_string())
"""

TRACK = """
row = tracker.make_row(
    variant=VARIANT,
    views=VIEWS,
    facets=FACETS,
    facet_weight=FACET_WEIGHT,
    local_cv=local_cv,
    local_folds=local_folds,
    n_train=len(train),
    n_pool=len(pool),
    notes="__NOTES__",
)
tracker.append_csv(WORKING / "tracker.csv", row)

print("TRACKER_JSON " + tracker.to_json(row))
print()
print("After submitting, record the public score locally with:")
print(f"  uv run python scripts/track.py add --variant {VARIANT} "
      f"--views {','.join(VIEWS)} --facets {','.join(FACETS)} "
      f"--local-cv {0.0 if local_cv is None else local_cv:.4f} --public-lb <score>")
"""

FOOTER = """
## Reading the result

- **Local CV up, public LB up** — the facet family is real. Keep it, and try
  giving it more weight (``facet_weight`` > 1).
- **Local CV up, public LB flat or down** — the gain is prompt-specific. The
  block is fitting the 150 train prompts, not the authors. Drop the family.
- **Both flat** — the family is null on this corpus (structure and emoji are,
  because the passages contain no markdown or emoji). Record the null; do not
  re-run it.

Download `/kaggle/working/tracker.csv` after each run and merge it:

```bash
uv run python scripts/track.py merge ~/Downloads/tracker.csv
uv run python scripts/track.py report
```
"""


def _render(template: str, variant: dict) -> str:
    return (
        template.replace("__ID__", variant["id"])
        .replace("__TITLE__", variant["title"])
        .replace("__HYPOTHESIS__", variant["hypothesis"])
        .replace("__EXPECTED__", variant["expected"])
        .replace("__VIEWS__", repr(variant["views"]))
        .replace("__FACETS__", repr(variant["facets"]))
        .replace("__NOTES__", variant["notes"])
    )


def cells_for(variant: dict) -> list[dict]:
    """The markdown/code cells of one variant, in order (bundle excluded)."""
    return [
        md(_render(HEADER, variant)),
        code(SETUP),
        code(_render(CONFIG, variant)),
        code(LOAD),
        code(LOCAL_CV),
        code(SUBMIT),
        code(_render(TRACK, variant)),
        md(FOOTER),
    ]


def path_for(variant: dict) -> Path:
    return NOTEBOOKS / f"04_{variant['slug']}.ipynb"


def build(variant: dict) -> dict:
    cells = []
    for i, cell in enumerate(cells_for(variant)):
        cell = dict(cell)
        cell.setdefault("id", f"{variant['id']}-cell-{i:03d}")
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        out = path_for(variant)
        out.write_text(json.dumps(build(variant), indent=1, ensure_ascii=False) + "\n")
        print(f"Wrote {out}  ({len(cells_for(variant))} cells)")
    print("\nNext: uv run python scripts/sync_notebook.py --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

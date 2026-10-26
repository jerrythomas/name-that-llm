# Name That LLM — Kaggle competition scaffold

[![CI](https://github.com/jerrythomas/name-that-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/jerrythomas/name-that-llm/actions/workflows/ci.yml)

Identify which of **18 frontier LLMs** wrote a given passage (authorship
attribution). Evaluated on **balanced accuracy** (mean per-class recall).

## Competition facts

| | |
|---|---|
| Task | 18-way text classification of LLM-authored passages |
| Train | `train.csv` — 2,699 passages / 150 prompts (`id, prompt_id, text, model`) |
| Test | `test.csv` — ~4,000 hidden passages; prompts unseen in train |
| Passages | 40–250 words; philosophy, physics, math, biology, CS, creative prose |
| Metric | Balanced accuracy |
| Output | `submission.parquet` with `id, model` (code competition) |

Valid labels: see `ntl.config.LABELS` (5 OpenAI, 4 Google, 5 Anthropic, 1 Qwen,
2 ZAI, 1 Moonshot).

## Setup

```bash
uv sync --extra dev      # creates .venv, installs ntl (editable) + deps
```

The package lives in `src/ntl` and is installed editable, so notebooks can
`import ntl` directly.

## Running standalone on Kaggle

`notebooks/01_eda.ipynb`, `notebooks/02_style_and_structure.ipynb`,
`notebooks/03_facets.ipynb` and the five `notebooks/04_v*.ipynb` submission
variants are **fully self-contained**: each embeds the whole `ntl` package in
an auto-generated bootstrap cell, so on Kaggle they need no `pip install`, no
`uv`, and no internet — just upload the `.ipynb` as a Code Competition
notebook.

The modular source stays the single source of truth. After editing
`src/ntl/`, regenerate the notebooks' inline cells:

```bash
uv run python scripts/sync_notebook.py --all   # every numbered notebook
uv run python scripts/sync_notebook.py --notebook notebooks/02_style_and_structure.ipynb
```

`sync_notebook.py` materialises the *real* package in memory
(`ntl` + `ntl.config`, `ntl.data`, …), so notebook code keeps using
`from ntl import data, explore, ...` unchanged — identical locally and on Kaggle.

Both notebooks print where they expect the data to be *before* they load it,
so a Kaggle run that cannot find the competition files shows the mount layout
in its output instead of failing blind.

Alternatively, produce a flat single-file bundle for use via `%run` or an
attached Kaggle Dataset:

```bash
uv run python scripts/build_bundle.py      # -> dist/ntl_bundle.py
```

## Get the data

1. Accept the competition rules on Kaggle.
2. Put API credentials at `~/.kaggle/kaggle.json`.
3. Download into `data/raw/`:

```bash
uv run python scripts/download_data.py
```

`ntl` resolves files from `data/raw/` locally.  On Kaggle it **discovers**
the competition directory inside the mount rather than assuming a layout —
Kaggle has used several, and none of them is guaranteed:

```
/kaggle/input/competitions/name-that-llm/     # classic
/kaggle/input/name-that-llm/                   # plain
/kaggle/input/datasets/<owner>/name-that-llm/  # current
```

`ntl.config.competition_dirs()` walks the mount (bounded depth) and ranks
every directory that holds `train.csv` / `test.csv` /
`sample_submission.parquet`, preferring the one that holds all three — so an
attached public passage pool that happens to ship its own `train.csv` cannot
displace the real competition data.  If nothing is found, the error message
prints the mount tree, so a missing attachment is diagnosable from the
notebook output alone.

```python
from ntl import data

print(data.data_locations())   # competition dirs, attached datasets, local dir
print(data.mount_summary())    # a readable dump of /kaggle/input
```

## Project layout

```
src/ntl/
  config.py    # paths, mount discovery, 18 labels, provider map, constants
  data.py      # load_train / load_test / inventory / in-style pool loader
  features.py  # cheap stylistic features (punct, markdown, lexical)
  views.py     # linguistic views: char / word / shape / funcon / punct
  facets.py    # 52 response + 8 group-relative style facets, by family
  measure.py   # within-group η², MI, per-family grouped CV, domain probe
  stylometry.py# log-odds verbal fingerprints (Monroe et al. 2008)
  groups.py    # recover prompt groups from text alone (k-NN + components)
  decision.py  # decision rules: argmax, Sinkhorn, Hungarian assignment
  pipeline.py  # the stack: views + facets -> centring -> classifier -> assignment
  tracker.py   # experiment log: local CV vs public LB, config-hashed
  explore.py   # EDA: balance, prompts, per-model stats, fingerprints
  plots.py     # charts (return fig/ax, optional save to outputs/)
  baseline.py  # TF-IDF (word + char) + logistic regression, random & grouped CV
  submit.py    # build / validate / write submission.parquet
notebooks/01_eda.ipynb              # end-to-end walkthrough (self-contained)
notebooks/02_style_and_structure.ipynb  # the hypothesis-driven exploration
notebooks/03_facets.ipynb           # which facets survive prompt control
notebooks/04_v{1..5}_*.ipynb        # five submission variants (one per hypothesis)
metrics/tracker.csv        # local CV vs public LB, one row per run
docs/HYPOTHESIS.md         # the facet programme: hypotheses, constraints, tracker
scripts/download_data.py   # kaggle CLI download + unzip
scripts/build_bundle.py    # -> dist/ntl_bundle.py (flat offline bundle)
scripts/sync_notebook.py   # inline src/ntl into a notebook's bootstrap cell
scripts/make_notebook.py   # (re)generate notebooks/02 from plain text
scripts/make_facets_notebook.py  # (re)generate notebooks/03
scripts/make_variants.py   # (re)generate the five submission variants
scripts/track.py           # add / merge / report experiment rows
scripts/run_notebook.py    # execute a notebook and (optionally) save outputs
scripts/check_notebooks.py # assert notebooks match their sources
scripts/selftest.py        # fast invariant checks (no heavy training)
```

```bash
uv run python scripts/selftest.py     # 21 checks, a few seconds
```

## Runbook on Kaggle

1. **Open the notebook from the competition page** — *Code → Add Code → Your
   Script* (or import the `.ipynb` into a notebook that is *attached to the
   competition*).  A generic notebook has no competition data mounted, and no
   path-guessing can conjure it.
2. Upload `notebooks/02_style_and_structure.ipynb`.  It needs no
   `pip install` and no internet: the whole `ntl` package is inlined in cell 2.
3. **Run All.**  The first cell prints where the loader expects the data and
   dumps the mount layout, so a missing attachment is obvious immediately:

   ```
   data locations:
     competition_dirs      ['/kaggle/input/datasets/...']
     attached_datasets     ['/kaggle/input/datasets/prvsiyan/...']
     local_data_dir        /kaggle/working/data/raw
   mount tree:
     datasets/
       competitions/
         name-that-llm/
   ```
4. Attach the public in-style passage pool as an *Add Input* dataset and
   re-run — the loader picks it up automatically:

   ```python
   from ntl import data
   pool = data.load_in_style_pool()
   print(data.describe_in_style_pool(pool))   # rows, per-model counts, coverage
   ```
5. Write the submission with `ntl.submit` (columns `id, model`, parquet).
6. To run the variant study, upload `notebooks/04_v1_control.ipynb` …
   `notebooks/04_v5_full.ipynb` as five separate notebooks, attach the data and
   the pool to each, and run. Each writes its own `submission.parquet` and
   appends to `/kaggle/working/tracker.csv`; merge the CSV home with
   `scripts/track.py merge` and compare with `scripts/track.py report`.

## Repository and CI

The repository holds the package code and the notebook builder — **not** the
Kaggle data and **not** the sample notebook.  The data directory is
git-ignored and fetched with `scripts/download_data.py`; `dist/` is a build
artefact.

`.github/workflows/ci.yml` runs on every push and pull request:

1. `ruff check` + `ruff format --check`
2. `scripts/selftest.py` — 21 invariant checks
3. `scripts/check_notebooks.py` — asserts that every generated notebook
   (02, 03, the five variants) matches its builder cell-for-cell, and that the
   package inlined in every notebook is byte-identical to `src/ntl`
4. `scripts/build_bundle.py` — the flat bundle still builds

CI is pure verification: it never rewrites the notebooks, because
regenerating them would strip the stored outputs.  A red build means the
notebooks and their sources have drifted — run `make notebook` locally and
commit the result.

The same steps are available locally:

```bash
make install     # uv sync --extra dev
make check       # lint + self-test + notebook/bundle consistency (verification only)
make run-02      # execute notebook 02 and save the outputs
make run-03      # execute notebook 03 and save the outputs
make track       # show local CV vs public LB per variant
```

## Evaluation protocol (read this first)

Always cross-validate with `GroupKFold(prompt_id)`.  A **random** split is not
a leaky-but-optimistic estimate here — it is a *different and much harder*
task, and it scores 3–4× lower:

| protocol | word 1-2 grams | char 2-5 grams |
|----------|----------------|-----------------|
| random stratified | 0.077 | 0.116 |
| **grouped by `prompt_id`** | **0.272** | **0.309** |

The reason is label noise, not leakage: every prompt is answered by all 18
models, so a random split puts ~13.6 of each validation passage's siblings
into the training fold, each with a **different** label.  The classifier
memorises the training fold perfectly either way (train accuracy 1.000),
but under a random split what it memorises is *"this content → these 18
different authors"*, so the prediction is close to a coin flip.  The
leaderboard condition — train on prompts, predict unseen prompts — is the
grouped one.

## The stack in one paragraph

Every passage answers a prompt, so most of a passage's surface form is the
prompt's, not the model's.  The pipeline therefore (1) represents passages
in several **views** — raw character/word n-grams plus *topic-free* views
(function words, word shape, punctuation) — (2) **recovers the prompt
groups** from text alone with a k-NN graph, (3) **subtracts each group's
mean** so that what is left is the author's residual, (4) classifies with a
balanced logistic regression, and (5) **assigns labels jointly inside each
group** (Hungarian), using the fact that a prompt is answered by each model
once.

Measured under prompt-grouped cross-validation (the honest protocol):

| step | balanced accuracy |
|------|-------------------|
| char n-grams only | 0.309 |
| + word & function-word views | 0.365 |
| + prompt centring | 0.425 |
| + group assignment | **0.467** |

(The per-fold spread is ±0.04 on 2,699 passages, so treat differences below
~0.01 as noise.  All figures are pooled prompt-grouped CV scores.)

The **facet block** is a parallel ladder on the same folds (notebook 03, 3-fold
screen so the five variants are comparable to each other):

| step | balanced accuracy |
|------|-------------------|
| char + word + funcon (control) | 0.446 |
| + punctuation | 0.433 |
| + voice | 0.439 |
| + voice + punctuation | 0.473 |
| + voice + group deviation | 0.472 |
| + every response facet + group deviation | **0.490** |

The interesting result is that the facets are **not additive**: no single
family beats the control on the full stack, but the pairs do, and the union
does best. Punctuation alone scores 0.245 in grouped CV and the group-deviation
family 0.186, so both carry real signal — they are just redundant with the
char/word views until a voice block makes them usable. Structure, emoji and
citation are null on this corpus (the passages contain no markdown or emoji).
The full measurement is in `notebooks/03_facets.ipynb`.

The noise floor matters when reading the notebooks: the layer-by-layer gains
(0.06, 0.06, 0.04) are well above it, but the *choice of grouping cut* is
decided by which cut actually recovers the prompts — not by the downstream
score, which barely moves.

## Usage in a notebook

```python
from ntl import data, explore, plots, features, baseline, submit
from ntl.config import LABELS

train, test = data.load_train(), data.load_test()
dist = explore.label_distribution(train)
plots.plot_label_distribution(dist)

preds = baseline.fit_predict(train, test, baseline.build_char_pipeline())
sub = submit.make_submission(test, preds)
submit.write_submission(sub, "submission.parquet")
```

…or the full stack:

```python
from ntl import pipeline

cfg = pipeline.StackConfig(views=("char", "word", "funcon"))
model = pipeline.StackModel(cfg)
model.fit(train["text"], train["model"], groups=train["prompt_id"])
preds = model.predict(test["text"])
```

## Running notebooks

```bash
uv run jupyter lab        # then open notebooks/01_eda.ipynb
```

## Tracking submissions

Each variant notebook prints a `TRACKER_JSON` line and appends a row to
`/kaggle/working/tracker.csv`.  Bring it home and read the table:

```bash
uv run python scripts/track.py merge ~/Downloads/tracker.csv
uv run python scripts/track.py report
```

`report` keeps the latest run per variant and shows `gap = public_lb -
local_cv`.  A positive gap means the pool (or the hidden prompts) helped; a
negative gap means the local CV was optimistic.  `metrics/tracker.csv` is the
committed record.

## Roadmap

> The facet programme — every candidate style metric, the prompt-side
> constraint that rules half of them out at test time, the five submission
> variants, and the local-vs-public tracker — is specified in
> [`docs/HYPOTHESIS.md`](docs/HYPOTHESIS.md). Update its experiment register
> after every submission.

1. ~~Group CV by `prompt_id` for an honest unseen-prompt estimate.~~ done
2. ~~Topic-free views + prompt centring + group assignment.~~ done
3. ~~Measure every response facet, select on within-group η², and screen
   combinations.~~ done — `notebooks/03_facets.ipynb`; facets add **+0.043** to
   the full stack (0.446 → 0.490 on three folds)
4. ~~Ship five submission variants and a local-vs-LB tracker.~~ done —
   `notebooks/04_v{1..5}_*.ipynb`, `ntl.tracker`, `scripts/track.py`
5. **Run the five variants on Kaggle and record the gap.**  This is the next
   measurement: five rows, one public-LB column, one interpretation per row.
6. Attach the public *in-style passage* pool (~40k passages, e.g.
   `prvsiyan/ntl-public-in-style-pool-v1`) and retrain the same stack on it —
   the largest known lever (0.563 → 0.604 in the public lineage).  The loader
   plumbing is in place and now filters the pool safely:

   ```python
   from ntl import data

   print(data.data_locations())           # where the loader thinks things are
   pool = data.load_in_style_pool()       # every mounted dataset, competition excluded
   print(data.describe_in_style_pool(pool))  # size, per-model counts, label coverage
   pool = data.prepare_training_pool(pool, train_texts=train["text"])
   ```

   `prepare_training_pool` keeps only the 18 labels, drops train duplicates
   and namespaces pool prompt ids — three silent failure modes otherwise.
7. Add frozen-LLM-embedding features (register that n-grams miss).
8. Check the recovered groups on the real test pool; soften the assignment
   step if the hidden test is not one passage per model per prompt.
9. Turn the log-odds fingerprints into explicit features.

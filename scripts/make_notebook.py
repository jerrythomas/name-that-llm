"""Generate notebooks/02_style_and_structure.ipynb.

The notebook is the deliverable; this script is just a convenient way to
author it as plain text and keep the JSON tidy.  Run:

    uv run python scripts/make_notebook.py
    uv run python scripts/sync_notebook.py --notebook notebooks/02_style_and_structure.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "02_style_and_structure.ipynb"


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


CELLS: list[dict] = []


def add(*cells: dict) -> None:
    CELLS.extend(cells)


# ---------------------------------------------------------------------------
# 0. Framing
# ---------------------------------------------------------------------------

add(
    md(
        """
# 02 — Style & Structure: finding the author's fingerprint

*How do you tell which of 18 LLMs wrote a passage, when all 18 were asked
the same question?*

This notebook is an **exploration**, not a leaderboard submission.  Each
section follows the same rhythm:

> **Hypothesis** → **Step** → **Evidence** → **Conclusion**

The question that drives it is the one the competition blurb hints at and
that most people miss:

> The *content* of a passage is chosen by the prompt.  The *way it is
> written* is chosen by the model.  So identify the author by the writing,
> not by the subject matter.

Everything below is measured on the real training set with honest
cross-validation.  Numbers quoted in the prose are the ones the code in the
same section prints.

**What is already established** (from `01_eda.ipynb` and the companion
notebook *The Prompt Is Not The Author*):

* 2,699 train passages · 150 prompts · 18 models · exactly one passage per
  (prompt, model) pair, except one prompt with 17.
* The metric is **balanced accuracy** — the mean of per-class recalls.
* The hidden test set gives no `prompt_id`, and its class mix is *not*
  uniform, and differs between the public and private leaderboards.

**Runtime.**  This notebook runs ~45–60 minutes on the Kaggle CPU runtime
(it is a measurement document, so it measures rather than asserts).  The two
heaviest cells are the per-view comparison in §3 and the view sweep in §6;
both can be skipped without affecting the rest of the story.

**What this notebook establishes**

| # | Question | Answer |
|---|----------|--------|
| 1 | Can a random cross-validation score be trusted? | No — it is a *different task*. 0.077 vs 0.272. |
| 2 | Do the models have distinguishable verbal habits? | Yes at *lab* level (z ≈ +11), no at sibling level (z ≈ +3, topic only). |
| 3 | Which representation exposes the habit? | Topic-free views. Function words help; shape and punctuation help *alone* but hurt in the union. |
| 4 | How much of a passage is the prompt? | Almost all of it — and it can be subtracted. |
| 5 | Can the prompt groups be recovered without `prompt_id`? | Yes, at ARI 0.97 — but only with a cut tuned at *pool* scale. |
| 6 | What does balanced accuracy actually reward? | Plain argmax. Rebalancing tricks are assumptions, not consequences. |

The stack in one line:

```
views (char + word + function words) → subtract the group mean
   → logistic regression → Hungarian assignment inside each recovered group
```

which scores **0.467** under prompt-grouped cross-validation, against 0.309
for a plain char-n-gram baseline.  (Per-fold the score ranges from 0.42 to
0.53 on 2,699 passages, so treat differences below ~0.01 as noise — the
number to quote is the pooled one, and it is the one this notebook prints.)
"""
    ),
    code(
        """
import textwrap
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ntl import baseline, data, decision, explore, groups, pipeline, plots, submit
from ntl.config import LABELS, PROVIDERS

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_colwidth", 130)
pd.set_option("display.width", 160)

# --- Where is the data?  -----------------------------------------------
# Printed *before* loading, so that a missing or oddly-laid-out Kaggle
# mount shows up in the printed output and not only in a traceback.
locations = data.data_locations()
print("data locations:")
for key, value in locations.items():
    print(f"  {key:20s} {value}")
if Path("/kaggle/input").exists():
    print("mount tree:")
    for line in data.mount_summary():
        print("   ", line)
    if not locations["competition_dirs"]:
        print()
        print("!!  The competition data was NOT found in the mount.")
        print("    Open this notebook from the competition page (Add Code > Your")
        print("    Script) so Kaggle mounts the competition files, or attach the")
        print("    competition dataset from the right-hand panel, then re-run.")
print()

train = data.load_train()
train["n_words"] = train["text"].str.split().str.len()

texts = train["text"].to_numpy(dtype=object)
labels = train["model"].to_numpy(dtype=object)
truth = train["prompt_id"].to_numpy(dtype=object)
families = np.array([PROVIDERS[m] for m in labels])

print(f"train: {train.shape[0]} passages · {train['prompt_id'].nunique()} prompts · {train['model'].nunique()} models")
train.head(3)
"""
    ),
)

# ---------------------------------------------------------------------------
# 1. The evaluation trap
# ---------------------------------------------------------------------------

add(
    md(
        """
---
## 1. The evaluation trap

**Hypothesis.**  Every prompt is answered by all 18 models, so the training
set contains 18 near-synonymous passages per prompt.  A *random* split puts
siblings of every validation passage into the training fold — and every one
of those siblings carries a **different label**.  The classifier is then
asked to map a single piece of content to 18 different authors.  The
cheapest way to fit that is to ignore the content, and with it most of the
style.  So a random split does not merely *leak*; it quietly turns the task
into a different, much harder one.

**Step.**  Score the same two pipelines twice — once with a random
stratified split, once with `GroupKFold(prompt_id)`, which keeps all 18
answers to a prompt in the same fold — and look at what the fitted model
does with a validation passage.

**Evidence.**  Below.
"""
    ),
    code(
        """
rows = []
for name, pipe in (
    ("word 1-2 grams", baseline.build_word_pipeline()),
    ("char 2-5 grams", baseline.build_char_pipeline()),
):
    random_cv = baseline.cross_val_scores(train, pipe)["balanced_accuracy"]
    grouped_cv = baseline.grouped_cv_scores(train, pipe)["balanced_accuracy"]
    rows.append({"features": name, "random CV": random_cv, "group CV (prompt_id)": grouped_cv})

cv_table = pd.DataFrame(rows)
print(f"chance for 18 classes = {1/18:.4f}\\n")
cv_table
"""
    ),
    code(
        """
# Why?  Three diagnostics.
#   * how much of the training fold the classifier memorises
#   * how confident it is on a validation passage
#   * how many same-prompt siblings of a validation passage are in the
#     training fold — with a *different* label each
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def fit_and_report(train_idx, valid_idx):
    vec.fit(texts[train_idx])
    A = vec.transform(texts[train_idx])
    B = vec.transform(texts[valid_idx])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0).fit(A, labels[train_idx])
    return (
        clf.score(A, labels[train_idx]),
        clf.predict_proba(B).max(axis=1).mean(),
    )


diagnostics = []
for name, splitter, use_groups in (
    ("random stratified", StratifiedKFold(5, shuffle=True, random_state=0), False),
    ("grouped by prompt", GroupKFold(n_splits=5), True),
):
    train_acc, confidence = [], []
    for train_idx, valid_idx in splitter.split(texts, labels, truth if use_groups else None):
        a, c = fit_and_report(train_idx, valid_idx)
        train_acc.append(a)
        confidence.append(c)
    diagnostics.append(
        {
            "protocol": name,
            "train accuracy": round(float(np.mean(train_acc)), 4),
            "mean max posterior on validation": round(float(np.mean(confidence)), 4),
        }
    )

# sibling contamination of a random split
train_idx, valid_idx = next(iter(StratifiedKFold(5, shuffle=True, random_state=0).split(texts, labels)))
train_groups = truth[train_idx]
siblings = np.mean([(train_groups == truth[v]).sum() for v in valid_idx])

pd.DataFrame(diagnostics)
print(f"same-prompt siblings of each validation passage that sit in the")
print(f"training fold of a random split: {siblings:.2f}  (each with a DIFFERENT label)")
"""
    ),
    md(
        """
**Conclusion.**  The random split is not a noisy estimate of the real
score — it is a **different task**.  Word features score 0.077 under a
random split and 0.272 under a prompt-grouped split; character features
0.116 vs 0.309.  The diagnostics say why:

* the classifier **memorises the training fold perfectly in both cases**
  (train accuracy 1.000) — nothing is "leaked" in the usual sense;
* it is **equally unsure** in both cases (mean maximum posterior 0.118);
* but a random split puts **13.6 same-prompt siblings of a validation
  passage into the training fold**, each with a different label.

So what the model memorises differs.  Under a random split the memorised
mapping is *"this content → these 18 different authors"*, so when a new
answer arrives the content points at eighteen labels at once and the
prediction is close to a coin flip (0.077 against a chance of 0.056).
Under a grouped split the memorised mapping is *"this content → no
information at all"*, which leaves the model free to use style — and the
score triples.

Which protocol matches the leaderboard?  The leaderboard condition is:
train on the whole train set (all siblings included), predict passages
answering **unseen** prompts.  A test passage's siblings are therefore
*not* in the training set — precisely the grouped condition.  So from here
on, **every** number is grouped by `prompt_id`.
"""
    ),
)

# ---------------------------------------------------------------------------
# 2. The verbal fingerprint
# ---------------------------------------------------------------------------

add(
    md(
        """
---
## 2. Do the models have verbal habits?  (the question, taken literally)

**Hypothesis.**  If a model has a *style*, it should have words it reaches
for and words it avoids.  The right statistical question is not "which words
are frequent" (that measures length and topic) but

> *given that this word is used at all, how much more does model A use it
> than the other seventeen?*

which is the Monroe et al. (2008) **log-odds ratio with an informative
Dirichlet prior**.  Its z-score is comparable across models, so the
fingerprints can be ranked against each other.

**Step.**  Compute the one-vs-rest log-odds z-score of every content word
for every model, and the same statistic for every *lab* (a lab pools all of
its models).

**Evidence.**
"""
    ),
    code(
        """
fingerprints = explore.log_odds_fingerprints(train, top_n=12, min_count=20)

for model in ("gemini-3.1-pro-preview", "gpt-5.5", "claude-opus-5"):
    d = fingerprints[model]
    words = ", ".join(f"{w} ({z:+.0f})" for w, z in zip(d["word"], d["z"]))
    print(f"{model:24s} {words}")
"""
    ),
    code(
        """
family_profiles = explore.family_fingerprints(train, top_n=14, min_count=20)

for family, d in family_profiles.items():
    words = ", ".join(f"{w} ({z:+.0f})" for w, z in zip(d["word"], d["z"]))
    print(f"{family:12s} {words}")
"""
    ),
    code(
        """
plots.plot_family_fingerprints(family_profiles)
"""
    ),
    code(
        """
plots.plot_log_odds_fingerprints(fingerprints)
"""
    ),
    code(
        """
# The other half of "verbal tics": the phrases a model repeats, and the way
# it *opens* its sentences.  Openers are almost topic-free — "It is worth
# noting that…", "There is…", "Let me…" — so they are a purer style signal
# than the words themselves.
for model in ("gemini-3.1-pro-preview", "gpt-5.5", "claude-opus-5"):
    phrases = explore.repeated_phrases(train, model, top_n=8, n=2)
    print(f"{model}")
    print("   repeated: " + ", ".join(f"'{p}' x{c}" for p, c in zip(phrases["phrase"], phrases["count"])))
    openers = explore.sentence_openers(train, model, top_n=8, n_words=3)
    print("   openers:  " + ", ".join(f"'{o}' x{c}" for o, c in zip(openers["opener"], openers["count"])))
    print()
"""
    ),
    md(
        """
**Evidence (read the two tables together).**

* At the **lab** level the fingerprints are strong and legible: Google's
  models reach for *profound, absolute, void, abyss, microscopic, cosmic*
  (z up to **+11**); OpenAI's for *make, familiar, seem, turn, vanished*
  (+8); Anthropic's for *years, something, nothing, consider, itself*.
  These are register differences, and they are real.
* At the **model** level, inside a lab, the picture collapses.  Compare
  `gpt-5.5` (*like, warm, braided, vanishing, maps*) with its siblings:
  those words are *topic* words — the prompt was about a vanishing something
  — not habits.  Their z-scores are ≈ +3…+5, an order of magnitude weaker
  than the lab-level effect, and they change with the prompt.
* The **repeated phrases** and the **sentence openers** tell the same story
  more concretely.  A model that repeats "in a sense" or opens with "There
  is a" is exhibiting a habit; a model whose top phrases are content words
  ("molecular structure", "ancient civilization") is just answering the
  question.

**Conclusion.**  Vocabulary identifies the **lab**, not the **model**.  The
word "cathedral" in a `claude-opus-4-6` passage is worth almost nothing: it
tells us the prompt was about architecture, and every model answering that
prompt said something about cathedrals.  So a bag-of-words model is mostly
measuring the prompt — which brings us back to the problem the prompt
creates.
"""
    ),
)

# ---------------------------------------------------------------------------
# 3. Views
# ---------------------------------------------------------------------------

add(
    md(
        """
---
## 3. Removing the topic leaves the style

**Hypothesis.**  If vocabulary is mostly prompt, then a representation that
*deletes* topic should isolate style.  Build several **views** of the same
passage, each throwing away a different layer:

| view | what it keeps | what it throws away |
|------|---------------|---------------------|
| `char` | characters 2–5-grams (words, spacing, punctuation) | nothing |
| `word` | word 1–2-grams | nothing |
| `shape` | word *shape* only (`X`/`x`) — capitalisation habits | the words themselves |
| `funcon` | function words + punctuation; content words → `CONTENT` | the topic |
| `punct` | punctuation and whitespace only | every word |

`funcon` is the interesting one: it is the **syntactic skeleton** of the
passage.  Two answers to two different prompts that argue the same way
("however, … therefore, … it is …") look identical under `funcon`.

**Step.**  Score each view alone, twice: once for the 18-way model task and
once for the 6-way *lab* task.  If the hypothesis is right, the topic-free
views should close much of the gap between the 6-way and the 18-way score.

**Evidence.**  First, what the views actually do to one passage:
"""
    ),
    code(
        """
example = explore.view_examples(train.loc[0, "text"])
plots.plot_view_examples(example)
"""
    ),
    code(
        """
def view_score(view: str, target: str = "model", n_splits: int = 5) -> float:
    \"\"\"Group-CV balanced accuracy for one view, without centring or assignment.\"\"\"
    y = labels if target == "model" else families
    cfg = pipeline.StackConfig(views=(view,), center=False, assign=False)
    return pipeline.group_cv_score(texts, y, truth, cfg, n_splits=n_splits)["balanced_accuracy"]


rows = []
for view in ("char", "word", "shape", "funcon", "punct"):
    rows.append(
        {
            "view": view,
            "18-way (model)": view_score(view, "model"),
            "6-way (lab)": view_score(view, "lab"),
        }
    )
    print(rows[-1])

view_table = pd.DataFrame(rows)
print(f"\\nchance: 18-way {1/18:.4f} · 6-way {1/6:.4f}")
view_table
"""
    ),
    md(
        """
**Evidence.**  Every view finds the lab far more easily than the model:

* `char` → 0.309 for the model, 0.530 for the lab.
* `funcon` → 0.301 for the model, **0.541** for the lab — the *best lab
  detector per unit of model signal*, and it is the cheapest view to
  compute.
* `punct` → 0.194 / 0.433: pure formatting habits still identify the lab
  about 2.6× better than chance.

The gap between the two columns **is** the sibling-discrimination problem.
The topic-free views are not worse at the lab task than the rich ones — they
are simply *less distracted* by the prompt.

**Conclusion.**  Two actions follow.  (i) Build the model on a **union of
views** rather than one, so that lexical and structural evidence combine.
(ii) Never trust a representation that can only see the topic — the prompt
is the nuisance variable and must be handled explicitly (next).
"""
    ),
)

# ---------------------------------------------------------------------------
# 4. The prompt is not the author
# ---------------------------------------------------------------------------

add(
    md(
        """
---
## 4. The prompt is not the author

**Hypothesis.**  The 18 answers to one prompt are near-synonymous: same
subject, same facts, same framing.  What differs between them is *how* each
model says it.  If the shared part is estimated and removed, what is left is
the author's contribution.

**Step.**  Measure how much two passages that answered the same prompt have
in common, compared with two passages that did not.

**Evidence.**
"""
    ),
    code(
        """
D = groups.content_distance_matrix(texts)

same_mask = truth[:, None] == truth[None, :]
np.fill_diagonal(same_mask, False)

same_d, diff_d = D[same_mask], D[~same_mask]
order = np.argsort(D, axis=1)
nn_same = np.mean([truth[i] == truth[order[i, 0]] for i in range(len(truth))])

print(f"content-word cosine distance")
print(f"  same prompt      mean {same_d.mean():.3f}   median {np.median(same_d):.3f}")
print(f"  different prompt mean {diff_d.mean():.3f}   median {np.median(diff_d):.3f}")
print(f"  passages whose NEAREST NEIGHBOUR shares their prompt: {nn_same:.3f}")
"""
    ),
    md(
        """
**Evidence.**  Same-prompt passages sit at cosine distance 0.785 from each
other; different-prompt passages at 0.989.  And 99.8% of passages have a
nearest neighbour that answered the *same* prompt.

**Conclusion.**  The prompt structure is fully present in the texts, even
without `prompt_id`.  That is the lever: find those groups, and remove what
they have in common.
"""
    ),
)

# ---------------------------------------------------------------------------
# 5. Recovering the groups
# ---------------------------------------------------------------------------

add(
    md(
        """
---
## 5. Recovering the prompt groups without `prompt_id`

**Step.**  Build a k-nearest-neighbour graph over the content-word distance
matrix, keep edges below a cut, and take connected components.  A passage's
group is then the set of passages that answered its prompt.

**Evidence.**
"""
    ),
    code(
        """
pool_groups = groups.infer_groups(texts, k=3, cut=0.88)
report = groups.group_report(pool_groups, truth)
print(report)

plots.plot_group_sizes(groups.group_sizes(pool_groups), expected=18)
"""
    ),
    md(
        """
### 5.1 The cut has to be tuned at the scale it will be used

**Hypothesis.**  Connected components can *chain*: one spuriously close pair
of passages links two different prompts, and the whole cluster merges.  The
bigger the pool, the more chances there are for such a bridge — so a cut
tuned on a cross-validation fold (≈540 passages) may not survive the hidden
test pool (≈4,000 passages).

**Step.**  Measure the ARI of the recovered grouping as the pool grows, for
two different cuts.

**Evidence.**
"""
    ),
    code(
        """
prompts = np.unique(truth)
rng = np.random.default_rng(0)

rows = []
for n_prompts in (10, 25, 50, 100, 150):
    chosen = rng.choice(prompts, size=n_prompts, replace=False)
    mask = np.isin(truth, chosen)
    D_sub = D[np.ix_(mask, mask)]
    truth_sub = truth[mask]
    for cut in (0.88, 0.95):
        g = groups.infer_groups(texts[mask], k=3, cut=cut, distance=D_sub)
        rep = groups.group_report(g, truth_sub)
        rows.append(
            {
                "prompts in pool": n_prompts,
                "passages": int(mask.sum()),
                "cut": cut,
                "recovered groups": rep["n_groups"],
                "largest group": rep["largest_group"],
                "ARI": rep["ari"],
            }
        )

scale_table = pd.DataFrame(rows)
scale_table.pivot(index=["prompts in pool", "passages"], columns="cut", values="ARI")
"""
    ),
    code(
        """
# Does the downstream score actually care?  Measure the full stack with the
# scale-stable cut and with the chaining-prone one.
cut_rows = []
for cut in (0.88, 0.95):
    cfg = pipeline.StackConfig(
        views=("char", "word", "funcon"),
        center=True,
        assign=True,
        grouping_scope="pool",
        group_cut=cut,
    )
    score = pipeline.group_cv_score(texts, labels, truth, cfg, n_splits=5)[
        "balanced_accuracy"
    ]
    cut_rows.append({"group cut": cut, "balanced accuracy": score})
    print(cut_rows[-1], flush=True)

cut_comparison = pd.DataFrame(cut_rows)
"""
    ),
    md(
        """
**Evidence.**  With cut = 0.95 the grouping is *perfect* on a small pool
(ARI 1.000 at 10 and 25 prompts) and collapses at full size (ARI 0.58, one
cluster swallowing 252 passages).  With cut = 0.88 it is stable at every
size (ARI 0.96–0.98 from 180 to 2,699 passages).

**Conclusion.**  The neighbour relation itself is scale-free (99.8% at every
size) — it is the *clustering* step that was fragile.  Therefore:

1. use **cut = 0.88**, and
2. **evaluate the grouping over the whole pool**, never over a fold.

Point 2 changes the cross-validation protocol itself: the validation fold's
groups must be recovered from the *entire* pool, exactly as the test pool
will be.  `StackConfig.grouping_scope = "pool"` implements that.

An honest footnote: the *downstream* score is far less sensitive to the
grouping than the ARI suggests — the two numbers printed above differ by only
~0.01, barely above the noise floor, because the assignment step is
forgiving.  The score alone would not settle the choice.  But the right
choice is still the one that recovers the prompts, because the assignment
step is only as good as the group it is applied to, and the hidden pool is
larger than anything measured here.
"""
    ),
)

# ---------------------------------------------------------------------------
# 6. Subtracting the prompt, and the one-per-prompt constraint
# ---------------------------------------------------------------------------

add(
    md(
        """
---
## 6. Subtracting the prompt, and the one-per-prompt constraint

**Hypothesis.**  Two distinct pieces of structure are still unused.

* **Centring.**  The mean of the 18 answers to one prompt *is* the prompt,
  to a first approximation.  Subtracting it leaves the residual — what each
  model added.
* **Assignment.**  The train set has exactly one passage per (prompt, model).
  Inside a recovered group the 18 labels should therefore be a
  *permutation*, not 18 independent argmaxes.

**Step.**  Add the two layers to the stack and measure each separately.

**Evidence.**
"""
    ),
    code(
        """
rows = []
for label, views, center, assign in (
    ("char n-grams only", ("char",), False, False),
    ("+ word & function-word views", ("char", "word", "funcon"), False, False),
    ("+ prompt centring", ("char", "word", "funcon"), True, False),
    ("+ group assignment", ("char", "word", "funcon"), True, True),
):
    cfg = pipeline.StackConfig(views=views, center=center, assign=assign)
    score = pipeline.group_cv_score(texts, labels, truth, cfg, n_splits=5)[
        "balanced_accuracy"
    ]
    rows.append({"step": label, "balanced accuracy": score})
    print(rows[-1])

ablation = pd.DataFrame(rows)
"""
    ),
    code(
        """
plots.plot_ablation(
    list(zip(ablation["step"], ablation["balanced accuracy"])),
    baseline=1 / 18,
)
"""
    ),
    code(
        """
# Which views should the stack use?  (slow: a few minutes per configuration)
VIEW_SUBSETS = [
    ("char",),
    ("word",),
    ("funcon",),
    ("char", "word"),
    ("char", "funcon"),
    ("char", "word", "funcon"),
    ("char", "word", "funcon", "punct"),
    ("char", "word", "funcon", "shape"),
]

rows = []
for views in VIEW_SUBSETS:
    cfg = pipeline.StackConfig(views=views, center=True, assign=True)
    score = pipeline.group_cv_score(texts, labels, truth, cfg, n_splits=5)[
        "balanced_accuracy"
    ]
    rows.append({"views": "+".join(views), "balanced accuracy": score})
    print(rows[-1], flush=True)

view_sweep = pd.DataFrame(rows)
"""
    ),
    md(
        """
**Evidence (the view sweep).**  `char + word + funcon` is the best
combination; adding `punct` *hurts* (0.417 against 0.467) even though
punctuation alone is a decent lab detector — a reminder that a view which is
individually informative can still be noise once the other views are present.

**Conclusion.**  The layers are additive and each is worth a lot:

| step | balanced accuracy | gain |
|------|-------------------|------|
| char n-grams alone | 0.309 | — |
| + word & function-word views | 0.365 | +0.056 |
| + prompt centring | 0.425 | +0.060 |
| + group assignment | 0.467 | +0.042 |

(The gains are pooled over the five folds; the per-fold spread is ±0.04, so
read them as "each layer is worth several points", not as precise decimals.)

Two things are worth pointing at.

*The views and the centring are complementary, not redundant.*  The views
change *what* is measured; the centring removes *what the prompt decided*.
A richer view alone cannot get past the prompt, because the prompt decides
the topic, the vocabulary, and the framing.

*The assignment step is not a metric trick.*  It encodes a fact about how
the data was *made* (one passage per model per prompt), not an assumption
about the test class mix.  That distinction matters — see the next section.
"""
    ),
)

# ---------------------------------------------------------------------------
# 7. The metric
# ---------------------------------------------------------------------------

add(
    md(
        """
---
## 7. What does balanced accuracy actually reward?

The competition states that the number of test passages per model is **not**
equal, and that the public and private leaderboards have different mixes.
That makes any step which *balances the output* a bet on a distribution we
cannot observe.  Worth working out what the metric really asks for.

**The metric.**  With ``N_k`` test passages of true class ``k``::

    balanced_accuracy = (1/K) * Σ_k  #{i : y_i = k and ŷ_i = k} / N_k

Rewrite it as a sum over passages::

    = (1/K) * Σ_i Σ_k  P(y_i = k | x_i) * 1[ŷ_i = k] / N_k

Each passage can be labelled independently, and the expected contribution of
labelling passage *i* with class *j* is ``P(y_i = j | x_i) / N_j``.  So the
metric-optimal rule is::

    ŷ_i = argmax_j  P(y_i = j | x_i) / N_j

Now substitute the classifier's own posterior.  If it was trained with prior
``π_source`` and the test prior is ``π_test``, then
``P(y = j | x) ∝ p_source(j | x) · π_test_j / π_source_j``, while
``N_j ∝ π_test_j``.  **The prior terms cancel**::

    ŷ_i = argmax_j  p_source(j | x_i) / π_source_j

This train set is uniform (150 passages per model), so ``π_source`` is
uniform and the optimal rule is **plain argmax** — *whatever the test mix
happens to be*.

**Hypothesis.**  Two popular-looking corrections are therefore wrong:
re-balancing the output towards the observed (or towards uniform) class mix,
and dividing by an estimated test prior.

**Step.**  Simulate skewed test mixes by subsampling held-out passages, and
compare decision rules on each.

**Evidence.**
"""
    ),
    code(
        """
CLASSES = np.array(sorted(set(labels)))


def fit_predict_proba(train_idx, valid_idx):
    cfg = pipeline.StackConfig(views=("char",), center=False, assign=False)
    model = pipeline.StackModel(cfg)
    model.fit([texts[i] for i in train_idx], labels[train_idx], groups=truth[train_idx])
    scores, _ = model.decision_function([texts[i] for i in valid_idx])
    return scores


def make_pool(valid_idx, mode, seed=1):
    \"\"\"Subsample a fold to create a class mix (keeping the 1:1 group design).\"\"\"
    rng = np.random.default_rng(seed)
    by_class = {}
    for i in valid_idx:
        by_class.setdefault(labels[i], []).append(i)
    keep = []
    for cls, idxs in by_class.items():
        if mode == "uniform":
            take = idxs
        elif mode == "mild skew":
            take = idxs if cls in CLASSES[:3] else rng.choice(idxs, size=max(2, len(idxs) // 2), replace=False)
        else:
            take = idxs if cls in CLASSES[:2] else rng.choice(idxs, size=max(1, len(idxs) // 4), replace=False)
        keep.extend(list(take))
    return np.array(sorted(keep))
"""
    ),
    code(
        """
from sklearn.model_selection import GroupKFold

folds = list(GroupKFold(n_splits=5).split(texts, labels, truth))
rules = ("argmax", "rebalance→uniform", "rebalance→true mix", "divide by est. prior", "+ group assignment")
rows = []

for mode in ("uniform", "mild skew", "extreme skew"):
    acc = {name: [] for name in rules}
    counts = None
    for train_idx, valid_idx in folds:
        scores = fit_predict_proba(train_idx, valid_idx)
        position = {g: j for j, g in enumerate(valid_idx)}
        keep = np.array([position[g] for g in make_pool(valid_idx, mode)])
        S = scores[keep]
        y_true = labels[valid_idx][keep]
        g_true = truth[valid_idx][keep]
        true_counts = np.array([(y_true == c).sum() for c in CLASSES], float)
        counts = true_counts

        shifted = S - S.max(axis=1, keepdims=True)
        probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)

        acc["argmax"].append(decision.balanced_accuracy(y_true, CLASSES[S.argmax(1)]))
        acc["rebalance→uniform"].append(
            decision.balanced_accuracy(y_true, CLASSES[decision.sinkhorn(probs).argmax(1)])
        )
        acc["rebalance→true mix"].append(
            decision.balanced_accuracy(y_true, CLASSES[decision.sinkhorn(probs, true_counts).argmax(1)])
        )
        estimated = decision.em_prior(probs)
        acc["divide by est. prior"].append(
            decision.balanced_accuracy(y_true, CLASSES[(probs / np.maximum(estimated, 1e-9)).argmax(1)])
        )
        assigned = decision.assign_within_groups(S, g_true, CLASSES)
        acc["+ group assignment"].append(decision.balanced_accuracy(y_true, assigned))

    rows.append(
        {
            "test class mix": mode,
            "passages per class": f"{int(counts.min())}–{int(counts.max())}",
            **{name: round(float(np.mean(values)), 4) for name, values in acc.items()},
        }
    )

metric_table = pd.DataFrame(rows)
metric_table
"""
    ),
    md(
        """
**Evidence.**

*(These use the char-n-gram view on its own, so the absolute numbers are
the §1 baseline rather than the full stack — the point is the comparison
between rules, not the level.)*

* **Plain argmax** is stable across all three mixes (0.31 / 0.31 / 0.30) —
  exactly what the derivation predicts.
* **Rebalancing towards the true mix** is mildly *helpful* when the mix is
  uniform (0.32) and increasingly **harmful** as the skew grows
  (0.32 → 0.28 → 0.18).  Forcing the output to look like the class
  distribution does not maximise per-class recall.
* **Rebalancing towards a uniform output** is a small, mix-robust
  improvement (0.32 / 0.32 / 0.30) — the one heuristic that survives.
* **Dividing by an estimated test prior** is the wrong correction applied in
  the wrong direction, and it is catastrophic (0.10 — near chance).
* **Group assignment helps under every mix** (+0.11 uniform, +0.05 mild,
  +0.04 extreme), because the constraint "no model twice in a group"
  survives a skew that merely removes models from a group.

**Conclusion.**  The pipeline should **not** rebalance its output.  The
train prior being uniform is what makes plain argmax optimal, and the
balanced-accuracy metric never asked for a uniform *output* — it asked for
uniform *weight per class*.  (A mild rebalancing towards uniform measured
slightly positive and robust to the mix, so it is available as
`StackConfig.balance`, but it is off by default and it is a heuristic, not
a consequence of the metric.)

The **assignment** step, by contrast, is unaffected by this reasoning: it
encodes the *design* of the data (one passage per model per prompt), not a
belief about the class mix.  The two are easy to confuse, and confusing
them is how a pipeline ends up quietly assuming a test distribution it
cannot see.
"""
    ),
)

# ---------------------------------------------------------------------------
# 8. The stack
# ---------------------------------------------------------------------------

add(
    md(
        """
---
## 8. The stack, end to end

Putting the measured pieces together:

```
passages
   │
   ├─ view: char n-grams  ─┐
   ├─ view: word n-grams  ─┤  (union, each block L2-normalised)
   ├─ view: function words─┘
   │
   ├─ subtract the group mean      ← removes the prompt
   │
   ├─ logistic regression          ← 18-way, class_weight="balanced"
   │
   └─ Hungarian assignment inside each recovered group
                                    ← enforces one passage per model per prompt
```

**Evidence.**  The ablation from section 6, plus the honest pool-scope
grouping of section 5.
"""
    ),
    code(
        """
final_cfg = pipeline.StackConfig(
    views=("char", "word", "funcon"),
    center=True,
    assign=True,
    grouping_scope="pool",
)
final = pipeline.group_cv_score(texts, labels, truth, final_cfg, n_splits=5)
print(f"pooled group-CV balanced accuracy: {final['balanced_accuracy']:.4f}")
print(f"per fold: {[round(s, 4) for s in final['fold_scores']]}")
"""
    ),
    md(
        """
**How much of this number should you believe?**  The pooled score is the
right one to quote, but the *per-fold* spread printed above (≈0.42 to ≈0.53
on 2,699 passages) says the noise floor is roughly ±0.04.  So:

* differences of **0.01 or less** between configurations in this notebook
  are not evidence of anything — they are a dozen passages out of 2,699;
* the layer-by-layer gains in section 6 (0.06, 0.06, 0.04) are comfortably
  above that floor and are the real findings;
* the *choice of cut* (section 5.1) is **barely** settled by the downstream
  score — it is settled by which cut actually recovers the prompts, because
  the hidden pool is larger than anything measured here.

If you re-run this notebook with a different scikit-learn or scipy build,
expect the last decimal to move and the third to stay.
"""
    ),
)

# ---------------------------------------------------------------------------
# 9. Submission
# ---------------------------------------------------------------------------

add(
    md(
        """
---
## 9. Turning it into a submission — train on train + in-style pool

The single biggest lever in this competition is the public in-style passage
pool (~40k passages generated by the same 18 models under the same protocol).
Attaching it and retraining the same stack lifts the public-lineage score
from ~0.47 → ~0.60.

**On Kaggle:** *Add Input → Datasets → search "name-that-llm" → attach
`prvsiyan/ntl-public-in-style-pool-v1` (or similar).*

The cell below loads the pool if present, combines it with train, fits the
full stack, predicts on the test pool, and writes `submission.parquet`.
"""
    ),
    code(
        """
# 1. Load train + optional in-style pool
train = data.load_train()
pool = data.load_in_style_pool()
print(data.describe_in_style_pool(pool))  # rows, per-model counts, label_coverage

# 2. Combine
import numpy as np
if not pool.empty:
    texts = np.concatenate([train["text"].to_numpy(dtype=object), pool["text"].to_numpy(dtype=object)])
    labels = np.concatenate([train["model"].to_numpy(dtype=object), pool["model"].to_numpy(dtype=object)])
    group_ids = np.concatenate([train["prompt_id"].to_numpy(dtype=object), pool["prompt_id"].to_numpy(dtype=object)])
else:
    texts = train["text"].to_numpy(dtype=object)
    labels = train["model"].to_numpy(dtype=object)
    group_ids = train["prompt_id"].to_numpy(dtype=object)

print(f"Training on {len(texts)} passages ({len(train)} train + {len(pool)} pool)")

# 3. Fit final model on combined data
model = pipeline.StackModel(final_cfg)
model.fit(texts, labels, groups=group_ids)

# 4. Predict on test
test = data.load_test()
test_texts = test["text"].to_numpy(dtype=object)
predictions = model.predict(test_texts)

# 5. Write submission
submission = submit.make_submission(test, predictions)
problems = submit.validate_submission(submission, test["id"])
print("validation problems:", problems or "none")
submit.write_submission(submission, "/kaggle/working/submission.parquet")
print("Written: /kaggle/working/submission.parquet")
submission.head()
"""
    ),
    md(
        """
### 9.1 What the public test set can and cannot tell you

The public leaderboard's labels are **not** public, so there is no way to
compute the score locally — you submit, and read the leaderboard.  But the
public test *passages* are visible, and they are still useful, because the
pipeline is transductive:

* the **recovered group structure** can be inspected directly — how many
  groups, how big, how many passages per group;
* the **predicted class mix** can be compared with the published mix;
* and the *predicted* labels of a group can be checked for the property the
  pipeline assumes (at most one passage per model per group).

What none of that gives you is correctness.  A model can be confidently
wrong in a way that looks perfectly well-formed.  So the local diagnostics
below are a *smoke test*, not a validation.
"""
    ),
    code(
        """
# Diagnostics on the test pool — run this on Kaggle, where test.csv is the
# full hidden pool rather than the three local example rows.
test_groups = groups.infer_groups(test_texts, k=final_cfg.group_k, cut=final_cfg.group_cut)
sizes = groups.group_sizes(test_groups)
print(f"test passages        : {len(test_texts)}")
print(f"recovered groups     : {len(sizes)}")
print(f"group size (median)  : {int(np.median(sizes))}   (design size: 18)")
print(f"group size (min/max) : {int(sizes.min())} / {int(sizes.max())}")
print()
print("predicted class mix:")
print(submission["model"].value_counts().to_frame("n").assign(share=lambda d: d["n"] / d["n"].sum()).head(20))
"""
    ),
    code(
        """
# Write the submission file for the Kaggle grader
submit.write_submission(submission, "/kaggle/working/submission.parquet")
print("Written: /kaggle/working/submission.parquet")
"""
    ),
)

# ---------------------------------------------------------------------------
# 10. What is left
# ---------------------------------------------------------------------------

add(
    md(
        """
---
## 10. Where the remaining error is, and what to do next

**Honest accounting of the gap.**  The stack above reaches ≈0.47 balanced
accuracy under prompt-grouped cross-validation.  The strongest public
notebooks for this competition reach ≈0.60.  The difference is not a
better idea about *style* — it is more *data of the same kind* and richer
representations.  Concretely, the known levers:

| lever | why it should work | what it costs |
|-------|--------------------|---------------|
| **in-style passages** (public datasets of ~40k passages generated by the same models in the same protocol) | 15× more training data from exactly the right distribution; the single biggest jump in the known lineage (0.563 → 0.604) | attach a public Kaggle dataset; the loader (`ntl.data.load_in_style_pool`) is already written |
| **frozen LLM embeddings / stylometry features** (361 features in the known lineage) | captures register that n-grams miss | GPU + a download |
| **LoRA adapters per model** | a model that has *seen* the author's style will recognise it | GPU + training |

**What I would do next, in order.**

1. **Attach the in-style passage pool** and retrain the same stack on it.
   Everything in this notebook is orthogonal to that: the views, the
   centring, the grouping and the assignment all apply unchanged to a
   larger training set.  The loader is ready::

       from ntl import data
       print(data.data_locations())            # where the loader thinks things are
       pool = data.load_in_style_pool()        # every mounted dataset, competition excluded
       print(data.describe_in_style_pool(pool))  # size, per-model counts, coverage
2. **Check the recovered groups on the real test pool** (visible on Kaggle,
   invisible locally): plot the group-size histogram and the recovered
   group count.  If the hidden test is *not* one passage per model per
   prompt, the assignment step should be softened or dropped — the metric
   work in section 7 shows exactly how to measure the cost.
3. **Use the verbal fingerprints as features**, not just as a reading of the
   models: the log-odds z-scores of section 2 are a 18 × V table of
   "how much does model A like this word" and are cheap to compute at
   inference time from the *training* corpus.

**A last word on the original question.**  The models *do* have verbal
tics — but the tics that identify a **lab** are strong and legible
(*profound / absolute / void*), while the tics that separate **sibling
models** are mostly *structural* (syntax, formatting, sentence shape) and
barely lexical.  That is why the winning representation in this notebook is
not a word model but a *function-word* model, and why the prompt had to be
subtracted before any of it could be read.
"""
    ),
)


def main() -> int:
    cells = []
    for i, cell in enumerate(CELLS):
        cell = dict(cell)
        cell.setdefault("id", f"cell-{i:03d}")
        cells.append(cell)

    nb = {
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT}  ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

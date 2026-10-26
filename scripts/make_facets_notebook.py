#!/usr/bin/env python
"""Generate ``notebooks/03_facets.ipynb``.

The notebook is the deliverable.  This script is the tidy way to author it as
plain text.  Run:

    uv run python scripts/make_facets_notebook.py
    uv run python scripts/sync_notebook.py --notebook notebooks/03_facets.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "03_facets.ipynb"


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
# 0. The question
# ---------------------------------------------------------------------------

add(
    md(
        """
# 03 — Which facets actually identify an author?

Notebook 02 established that the pipeline works: **0.467** balanced accuracy
under prompt-grouped cross-validation, against 0.0556 chance. It also left a
question that a score cannot answer:

> *Is there a small, legible set of measurements — punctuation, length,
> vocabulary, formatting, deviation from the group — that carries the author
> signal, or is the whole gain hidden inside the n-gram views?*

That question matters for two reasons. If a facet family carries signal, it is
cheap to compute, easy to explain, and worth adding to the stack. If it only
*looks* informative, it will not transfer to a hidden prompt and will cost a
submission to find out.

## The trap this notebook is built around

A naive measurement says response length is a strong clue. Raw mutual
information with the model is **0.016** for word count and **0.085** for
average word length — both well above what 2,699 passages can produce by
chance. But word count is mostly a property of the *prompt*: some prompts ask
for an essay and some for a sentence. After prompt control, word count
collapses to η² **0.011**, while average word length holds at **0.39**. Same
statistic, opposite verdicts — that gap is the whole reason this notebook
exists.

The honest question is narrower:

> **After every prompt group is centred on its own mean, does the facet still
> vary with the model?**

That is a one-way ANOVA on the residuals, and it is the statistic this
notebook selects on (`eta2_within`). Raw mutual information is reported only
as a contrast — the distance between the two columns *is* the prompt effect.

## What the measurements say

| finding | evidence |
|---|---|
| Facets carry real author signal | punctuation alone scores **0.245** grouped CV (chance 0.056) |
| The em dash is the single best feature | within-group η² **0.40**, the highest of 60 facets |
| Response *length* is a prompt proxy; word *complexity* is not | word count η² **0.011**, average word length η² **0.39** |
| Deviation from the group is real | `g_centroid_cos` η² **0.249**; the group family alone scores 0.186 |
| Structure, emoji and citation are **null** on this corpus | at or within noise of chance in grouped CV |
| Facets add to the view stack | control 0.446 → **0.490** with all facets (3 folds) |
| Prompt-side facets cannot help | an inferred domain is constant inside 100% of recovered groups |

The last row is the constraint that shapes the whole competition: the test set
is `id, text` only, so a prompt-side facet is *unobservable at prediction
time*. The best it can do is stratify training. Anything that has to separate
siblings within a prompt must be a response facet or a group-relative one.
"""
    ),
    code(
        """
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ntl import data, facets, groups, measure, pipeline

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)

# The two heavy screens (sections 7 and 8) take about three minutes together.
RUN_HEAVY = True
SCREEN_FOLDS = 3

print("data locations:")
for name, paths in data.data_locations().items():
    print(f"  {name}: {paths or '(not found)'}")
"""
    ),
)

# ---------------------------------------------------------------------------
# 1. Load and recover the grouping
# ---------------------------------------------------------------------------

add(
    md(
        """
## 1. Load the corpus and recover the grouping

Every statistic below is within a prompt group, so the first job is to agree on
what a group is. Two groupings exist and both are printed:

* `prompt_id` — the truth, available on train only;
* `infer_groups(texts, k=3, cut=0.88)` — what the pipeline actually has at test
  time, recovered from content alone.

The recovered grouping is what matters for realism; the true ids are the
sanity check. On the full train pool the two agree at **ARI 0.967**.
"""
    ),
    code(
        """
train = data.load_train()
texts = data.as_text_array(train["text"])
y = data.as_text_array(train["model"])
true_groups = data.as_text_array(train["prompt_id"])

recovered = groups.infer_groups(texts, k=3, cut=0.88)
report = groups.group_report(recovered, true_groups)

print(f"passages           : {len(texts)}")
print(f"models             : {len(np.unique(y))}")
print(f"true prompt groups : {len(np.unique(true_groups))}")
print(f"recovered groups   : {len(np.unique(recovered))}")
print(f"ARI                : {report['ari']:.3f}")
print(f"purity             : {report['purity']:.3f}")
print()
print("passages per model:", dict(zip(*np.unique(y, return_counts=True), strict=True)))
print()
print("group sizes:", dict(zip(*np.unique(np.unique(recovered, return_counts=True)[1], return_counts=True), strict=True)))
"""
    ),
    md(
        """
**Reading the group sizes.** The modal group is exactly 18 — one answer from
each model — and 133 of the 157 recovered groups have that shape. The
imperfections are visible and worth naming: 10 singletons (a passage whose
prompt-mates grouped elsewhere), six groups of 17 (the one prompt that was
answered by only 17 models), four groups of 36 (two prompts bridged into one),
and a handful of fragments at 4, 14, 15 and 16. Centring skips groups outside
`[2, 18]`, so the 36s are left alone (their mean is two prompts, not one) and
the singletons are untouched. Assignment still gives every passage a label.
"""
    ),
)

# ---------------------------------------------------------------------------
# 2. The facet inventory
# ---------------------------------------------------------------------------

add(
    md(
        """
## 2. The facet inventory — and what is observable at prediction time

`ntl.facets` extracts **52 response facets** in nine families plus **8
group-relative facets**. Every one is a number computed from the passage
itself; none needs prompt text or a model call.

The observability column is the important one. `test.csv` is `id, text` — no
prompt, no `prompt_id`, no metadata. A response facet is available directly.
A group facet needs the *recovered* group, which is why the grouping step is
part of the model rather than an EDA convenience.
"""
    ),
    code(
        """
inventory = []
for family in [*facets.RESPONSE_FAMILIES, "group"]:
    columns = facets.facet_columns([family])
    inventory.append(
        {
            "family": family,
            "n_facets": len(columns),
            "examples": ", ".join(columns[:3]),
            "observable at test time": "needs recovered group" if family == "group" else "direct from text",
        }
    )
inventory = pd.DataFrame(inventory)
print(f"response facets: {len(facets.facet_columns(list(facets.RESPONSE_FAMILIES)))}")
print(f"group facets   : {len(facets.facet_columns(['group']))}")
print(f"total          : {len(facets.ALL_FACETS)}")
print()
inventory
"""
    ),
)

# ---------------------------------------------------------------------------
# 3. The trap
# ---------------------------------------------------------------------------

add(
    md(
        """
## 3. The trap: raw mutual information vs within-group η²

Two numbers per facet:

* **`mi`** — mutual information with the model label, computed over the whole
  corpus. High means "this facet and the author are associated", but the
  association can be entirely mediated by the prompt.
* **`eta2_within`** — the share of *within-group* variance explained by the
  model. This asks the question that transfers: **do siblings writing on the
  same prompt differ in this facet?**

A facet with high `mi` and near-zero `eta2_within` is a prompt proxy: it is
associated with the author only because it is associated with the prompt. A
facet with high `eta2_within` is an author habit. The selection rule is
`eta2_within`, never `mi` — and §5 shows why `mi` is worse than it looks.
"""
    ),
    code(
        """
signal = measure.facet_signal(texts, y, true_groups)
signal.round(4).head(20)
"""
    ),
    md(
        """
**What the table shows.**

* **The em dash is the strongest facet in the corpus** — within-group η² 0.40.
  This is the most legible result in the whole study: one model's house style
  is to punctuate with `—` where another uses `,` or `:`. It costs one
  number per passage.
* **Word length and long-word rate** (η² 0.39 / 0.38) are next: vocabulary
  complexity is an author property that survives the prompt.
* **Stopword rate** (0.30) and **contraction rate** (0.25) say the same thing
  from opposite directions: formal versus conversational register.
* **`g_centroid_cos` at 0.25** is the highest-ranked *relational* facet. It is
  not a property of the passage alone; it is how close the passage sits to the
  mean of its prompt group. That a relational feature ranks this high is
  direct evidence for the "the prompt is not the author" thesis.
* The two statistics disagree exactly where it matters. **Word count** has
  MI 0.016 but within-group η² 0.011 — a prompt proxy. **Average word length**
  has MI 0.085 and η² 0.39 — an author habit. And MI is **0.0** for the
  discrete person and opener facets even though they vary across groups, so it
  would have discarded real signal. Select on η², not on MI.
"""
    ),
    code(
        """
import matplotlib.pyplot as plt
import seaborn as sns

families = list(dict.fromkeys(signal["family"]))
palette = dict(zip(families, sns.color_palette("tab10", n_colors=len(families)), strict=False))

fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(
    data=signal,
    x="mi",
    y="eta2_within",
    hue="family",
    palette=palette,
    s=70,
    edgecolor="white",
    linewidth=0.8,
    ax=ax,
)
for row in signal.head(6).itertuples():
    ax.annotate(row.facet, (row.mi, row.eta2_within), fontsize=8, xytext=(4, 3), textcoords="offset points")
ax.axhline(measure.DEFAULT_SIGNAL_FLOOR, color="grey", linestyle="--", linewidth=1)
ax.text(0.01, measure.DEFAULT_SIGNAL_FLOOR + 0.008, "selection floor", color="grey", fontsize=8)
ax.set_xlabel("mutual information with the model (prompt included)")
ax.set_ylabel("within-group η² (prompt removed)")
ax.set_title("High MI and low η² = a prompt proxy; high η² = an author habit")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
The upper-right quadrant is the useful one: facets that are both common enough
to have mutual information and stable enough to survive prompt control. The
bottom-right is the trap — structure and citation have real `mi` and no
within-group signal at all.
"""
    ),
)

# ---------------------------------------------------------------------------
# 4. The dead families
# ---------------------------------------------------------------------------

add(
    md(
        """
## 4. The null result: structure, emoji and citation

Three families score **at or within noise of chance** in grouped CV —
structure 0.059, emoji 0.056, citation 0.056, against a chance level of
0.056. The reason is not that the probes are badly written; it is that this
corpus has been stripped of the things they measure. Eight of the ten
structure facets are constant (no markdown at all), the single emoji facet is
constant in 99.3% of groups, and all three citation facets are constant. The
two structure facets that do vary — digit rate and all-caps rate — carry
almost no usable signal.

This is worth stating plainly because the original brief asked whether models
format differently — bullet lists, tables, bold, code fences. **They do** in
the wild, but not in this corpus. The signal was removed upstream, so a
formatting submission would spend one of five daily submissions to measure
noise.
"""
    ),
    code(
        """
constants = (
    signal.loc[signal["constant_share"] > 0.5, ["facet", "family", "constant_share", "eta2_within", "mi"]]
    .sort_values("constant_share", ascending=False)
    .reset_index(drop=True)
)
print(f"{len(constants)} of {len(signal)} facets are constant in more than half of the groups")
constants.head(20)
"""
    ),
    code(
        """
dead = signal[signal["eta2_within"] < measure.DEFAULT_SIGNAL_FLOOR]
print(f"facets below the selection floor: {len(dead)} of {len(signal)}")
print()
print(dead.groupby("family").agg(n=("facet", "size"), max_eta2=("eta2_within", "max"), max_mi=("mi", "max")).round(4))
"""
    ),
    md(
        """
The pattern is family-wide, not facet-by-facet: eight of ten structure facets
are constant, the single emoji facet is constant in 99.3% of groups, and all
three citation facets are constant. A quarter of all 60 facets are constant in
more than half of the groups — mostly rare binary probes (question openers,
second person) that this corpus happens not to use. That is a property of the
corpus, and it is the kind of finding that is only possible *because* the
measurement is per-facet rather than per-score.
"""
    ),
)

# ---------------------------------------------------------------------------
# 5. Per-family classifier
# ---------------------------------------------------------------------------

add(
    md(
        """
## 5. Does the signal survive a classifier?

η² is a marginal statistic: it asks whether one facet separates authors, not
whether a family of facets can be combined into a decision. The end-to-end
check is a prompt-grouped cross-validation using **only** that family's
numbers — no n-grams, no content, no grouping tricks.

If a family scores near 0.0556 (chance), it is empty. If it scores well above,
the signal is real and combinable.
"""
    ),
    code(
        """
family_cv = measure.family_ranking(texts, y, true_groups, n_splits=5)
family_cv.round(4)
"""
    ),
    code(
        """
fig, ax = plt.subplots(figsize=(8, 4.5))
order = family_cv.sort_values("balanced_accuracy")
ax.barh(order["family"], order["balanced_accuracy"], color="#4c72b0")
ax.axvline(1 / 18, color="grey", linestyle="--", linewidth=1)
ax.text(1 / 18 + 0.002, -0.4, "chance", color="grey", fontsize=8)
for value, name in zip(order["balanced_accuracy"], order["family"], strict=False):
    ax.text(value + 0.003, name, f"{value:.3f}", va="center", fontsize=8)
ax.set_xlabel("prompt-grouped balanced accuracy (5 folds)")
ax.set_title("Facet families on their own — punctuation is the cheapest real signal")
ax.set_xlim(0, max(0.30, float(family_cv["balanced_accuracy"].max()) + 0.05))
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
**Punctuation 0.245, length 0.200, group 0.186, lexical 0.172.** A classifier
that sees nothing but twelve punctuation rates beats one that sees the entire
function-word vocabulary, which scores 0.338 in §8 — a quarter of the price for
a quarter of the score.

The ordering is also a warning about raw MI. It is not merely confounded by the
prompt; for discrete facets it is close to useless. `f_second_person` varies
across half the groups and has η² 0.035, yet its MI is exactly **0.0**, and so
is the MI of every opener facet. The kNN estimator behind
`mutual_info_classif` clips tied discrete values to zero. Selecting on MI would
have thrown away the person, hedging and opener families before they were ever
tested.
"""
    ),
)

# ---------------------------------------------------------------------------
# 6. Group-relative facets
# ---------------------------------------------------------------------------

add(
    md(
        """
## 6. Group-relative facets: deviation is a feature

Absolute rates answer "how does this model write?". The group facets answer a
different question: **"given this prompt, how does this model differ from its
siblings?"**

Eight numbers per passage, all computed after the grouping is recovered:
centroid cosine (how close to the group mean), length rank, length ratio,
word-count z-score, comma z-score, sentence z-score, and two rank variants.
The measurement is uneven: `g_centroid_cos` is strong (η² **0.249**) and
`g_comma_z` is real (0.190), while the length-relative facets sit near 0.01.
Even so the family scores **0.186** in grouped CV on its own — the relational
view pays for itself.
"""
    ),
    code(
        """
group_signal = signal[signal["family"] == "group"].sort_values("eta2_within", ascending=False)
group_signal.round(4)
"""
    ),
    code(
        """
frame = facets.facet_frame(texts, true_groups, ["group"])
frame = frame.assign(model=y)
by_model = frame.groupby("model")[["g_centroid_cos", "g_len_ratio", "g_word_rank", "g_comma_z"]].mean()
by_model.round(3).sort_values("g_centroid_cos")
"""
    ),
    md(
        """
**Reading the per-model table.** A model with a *low* mean centroid cosine is
one whose answers sit far from the group mean — a stylistic outlier. The
extremes are `gemini-3.8-flash` (0.357) and `claude-fable-5` (0.474); the
spread is narrow, which is itself informative: most models are trying to sound
similar, and the ones that don't are the easiest to identify.

`g_word_rank` is a normalised rank (0 = shortest, 1 = longest), so
`claude-opus-4-8` at 0.539 is the most consistently long-winded.
`g_len_ratio` is measured against the group **median**, not its mean: the
median ratio is 1.0 by construction and the average is 1.15 because the
distribution has a long right tail. Compare models with each other
(`gemini-3.8-flash` 1.007 vs `claude-opus-4-8` 1.236), not against 1.

This is the facet family that most directly implements the competition thesis.
It is also the one that requires the grouping to be right: at ARI 0.967 the
group mean is a good estimate of the prompt, but a merged group would average
two prompts and destroy the signal. That is why the grouping cut is tuned at
pool scale rather than fold scale (notebook 02, §7).
"""
    ),
)

# ---------------------------------------------------------------------------
# 7. The observability constraint
# ---------------------------------------------------------------------------

add(
    md(
        """
## 7. Why prompt-side facets cannot be used — a proof, not an opinion

The brief lists thirteen prompt axes: domain, clarity, few-shot-ness, question
count, depth, guardrails, prompt length, distance between prompt and response,
and so on. Every one is a property of the **prompt**, and the test set does
not contain prompts. Recovering the *group* tells you which passages share a
prompt; it does not tell you what the prompt was.

The consequence is stronger than "we don't have the text". A prompt-side
facet is **constant inside every recovered group**, so it cannot separate
siblings — which is precisely the job. We can demonstrate this by inferring a
domain label for each prompt from the group centroid and checking its
constancy.

If `constant_share == 1.0`, no prompt-side facet can contribute to the
within-group decision, no matter how good the inference is.
"""
    ),
    code(
        """
domains = measure.infer_domains(texts, true_groups, n_topics=6)
passage_domain = domains["passage_domain"].to_numpy()
share = measure.group_constant_share(passage_domain, true_groups)

print(f"topics/clusters        : {domains['n_topics']}")
print(f"groups with one domain : {share:.3f}")
print()
print("groups per domain:", domains["group_domain"].value_counts().sort_index().to_dict())
print()
print("A domain label is a property of the prompt, not of the author.")
print("It can stratify training; it can never be a within-group feature.")
"""
    ),
    md(
        """
**The constraint, stated once.** What is observable at prediction time is:
response facets, group-relative facets, and the recovered grouping itself.
What is not: prompt text, prompt id, prompt length, prompt domain, prompt
style, or the distance from response to prompt. Any proposal that depends on
the second list is untestable on the hidden set.

This does not make the prompt irrelevant. It makes the prompt a **nuisance
variable to be removed** (centring) rather than a **feature to be added**.
"""
    ),
)

# ---------------------------------------------------------------------------
# 8. Do facets add to the view stack?
# ---------------------------------------------------------------------------

add(
    md(
        """
## 8. Does the facet block add to the n-gram stack?

A family can be informative on its own and redundant with the views. The test
is a controlled comparison: same folds, same classifier, same centring, only
the facet block changes.

Two screens, both at 3 folds:

1. **function-word view only** — cheap, isolates the facet contribution;
2. **the full char + word + funcon stack** — the configuration that actually
   gets submitted.
"""
    ),
    code(
        """
VOICE = ("length", "lexical", "person", "hedging", "opener")
ALL = ("length", "lexical", "punctuation", "structure", "emoji", "person", "hedging", "citation", "opener", "group")

funcon_configs = [
    ("A  funcon only", dict(views=("funcon",), facets=())),
    ("B  funcon + punctuation", dict(views=("funcon",), facets=("punctuation",))),
    ("C  funcon + voice", dict(views=("funcon",), facets=VOICE)),
    ("D  funcon + group", dict(views=("funcon",), facets=("group",))),
    ("E  funcon + all facets", dict(views=("funcon",), facets=ALL)),
]

rows = []
for name, kwargs in funcon_configs:
    result = pipeline.group_cv_score(texts, y, true_groups, pipeline.StackConfig(**kwargs), n_splits=SCREEN_FOLDS)
    rows.append({"config": name, "balanced_accuracy": result["balanced_accuracy"], "fold_std": result["fold_std"]})
    print(f"{name:<28} {result['balanced_accuracy']:.4f}  (±{result['fold_std']:.3f})")

funcon_screen = pd.DataFrame(rows)
funcon_screen.round(4)
"""
    ),
    md(
        """
**On a function-word base the facets look additive.** The control scores 0.338;
punctuation alone adds +0.021, the voice block (without punctuation) +0.046, the
group block +0.039, and everything +0.102. The three contributions sum to 0.106
against a measured 0.102.

That looks like three independent, complementary signals. §8's second screen
shows that this is an artefact of the *weak* base: on the full char + word +
funcon stack, no single block adds anything on its own, and only the
combinations do. The cheap model is still a useful cost option (0.385 with one
view instead of three), but it is not a submission candidate.
"""
    ),
    code(
        """
if RUN_HEAVY:
    BASE = ("char", "word", "funcon")
    VOICE_NO_PUNCT = ("length", "lexical", "person", "hedging", "opener")
    VOICE_WITH_PUNCT = ("length", "lexical", "punctuation", "person", "hedging", "opener")

    full_configs = [
        ("F  base", dict(views=BASE, facets=())),
        ("G  base + punctuation", dict(views=BASE, facets=("punctuation",))),
        ("H  base + voice", dict(views=BASE, facets=VOICE_NO_PUNCT)),
        ("I  base + voice + punctuation", dict(views=BASE, facets=VOICE_WITH_PUNCT)),
        ("J  base + voice + group", dict(views=BASE, facets=VOICE_NO_PUNCT + ("group",))),
        ("K  base + all facets", dict(views=BASE, facets=ALL)),
    ]

    rows = []
    for name, kwargs in full_configs:
        result = pipeline.group_cv_score(texts, y, true_groups, pipeline.StackConfig(**kwargs), n_splits=SCREEN_FOLDS)
        rows.append({"config": name, "balanced_accuracy": result["balanced_accuracy"], "fold_std": result["fold_std"]})
        print(f"{name:<34} {result['balanced_accuracy']:.4f}  (±{result['fold_std']:.3f})")

    full_screen = pd.DataFrame(rows)
    full_screen["delta_vs_base"] = full_screen["balanced_accuracy"] - full_screen.loc[0, "balanced_accuracy"]
    full_screen.round(4)
else:
    full_screen = None
    print("skipped (RUN_HEAVY = False)")
"""
    ),
    md(
        """
**The full stack tells a different story: the facets are not additive.** On
identical folds:

* the control scores **0.4464**;
* punctuation alone **0.4327** (−0.014) and voice alone **0.4386** (−0.008) —
  both neutral, both inside the ±0.02 fold noise;
* voice **+ punctuation** scores **0.4730** (+0.027);
* voice **+ group deviation** scores **0.4720** (+0.026);
* everything scores **0.4897** (+0.043).

No single numeric block beats the control, yet the pairs do. The char view
already encodes punctuation, so adding it alone is redundant; the voice block
alone is too weak to matter. But punctuation gives the classifier a clean,
low-variance cue and voice gives it the register context that makes the cue
usable — the gain is an **interaction**, not a main effect.

This is why the five submitted variants are built around interactions rather
than a one-family-at-a-time ladder. It also means the honest claim is narrower
than "facets add +0.043 to the stack": facets add +0.043 **as a joint block**,
and the marginal value of any one family is close to zero.

Take the size of the effect seriously but not the decimal places. Three folds
of 150 prompts is a coarse instrument; the variant notebooks use the same three
folds precisely so that the five submissions are comparable to each other, not
to the five-fold 0.467 in notebook 02.
"""
    ),
)

# ---------------------------------------------------------------------------
# 9. The five variants
# ---------------------------------------------------------------------------

add(
    md(
        """
## 9. The five variants — what to submit, and why

The daily allowance is five submissions. The five variants are an
**interaction ladder**: v1 is the control, v2 tests whether the strongest
single family adds anything, v3 and v4 test the two interactions found in §8,
and v5 is their union.

| variant | views | facet block | hypothesis | local 3-fold |
|---|---|---|---|---|
| **v1** control | char + word + funcon | none | reference | 0.446 |
| **v2** punctuation | char + word + funcon | punctuation | does the strongest single family add on its own? | 0.433 |
| **v3** voice + punctuation | char + word + funcon | length, lexical, punctuation, person, hedging, opener | the first interaction | 0.473 |
| **v4** voice + deviation | char + word + funcon | length, lexical, person, hedging, opener, group | the relational interaction | 0.472 |
| **v5** full | char + word + funcon | all ten families | union of the interactions | 0.490 |

All five numbers are measured on the same three folds in §8, and each variant
notebook recomputes its own before submitting.

**Why this set?** Because §8 measured that single families are neutral on the
full stack. A ladder of one-family additions would have spent two submissions
(v2 and a one-view cheap model) on configurations that the local CV already
shows cannot beat the control. The two interaction variants are the ones with a
positive, reproducible local gain; v5 asks whether the union is worth more than
either pair.

**Why not a variant for structure/emoji?** Because §4 measured them as
constant. A submission is a scarce resource; spending one on a family that
cannot vary is not a test, it is a coin flip with extra steps. The null belongs
in the record (`docs/HYPOTHESIS.md` §9), not in the submission queue.

**What "gap" means.** Compare local CV with public LB per variant:

* local CV up and public LB up → the family transfers; keep it;
* local CV up and public LB flat → the family is fitting the 150 train
  prompts, not the authors; drop it;
* both flat → null on this corpus; record and move on.

The absolute public score is noisy at ~4,000 rows; the *ordering* of the five
variants is the measurement.
"""
    ),
    md(
        """
## 10. How to run and track the five variants

```bash
# generate and refresh the notebooks (already committed)
uv run python scripts/make_variants.py
uv run python scripts/sync_notebook.py --all

# local smoke test of any variant (3-row test stub; no real submission)
uv run python scripts/run_notebook.py notebooks/04_v5_full.ipynb
```

On Kaggle, upload `notebooks/04_v1_control.ipynb` … `04_v5_full.ipynb` as five
separate notebooks, attach the competition data and the in-style pool, and run.
Each writes `/kaggle/working/submission.parquet` and appends a row to
`/kaggle/working/tracker.csv`.

Bring the trackers home and read the table:

```bash
uv run python scripts/track.py merge ~/Downloads/tracker.csv
uv run python scripts/track.py report
```
"""
    ),
    code(
        """
# The tracker is just a CSV, so it is inspectable at any point.
from ntl import tracker

print("tracker columns:", tracker.TRACKER_COLUMNS)
print()
print("A variant row is hashed, so an edited notebook cannot inherit an old score:")
print(tracker.config_hash(
    ("char", "word", "funcon"),
    ("length", "lexical", "punctuation", "structure", "emoji", "person", "hedging", "citation", "opener", "group"),
))
"""
    ),
    md(
        """
## 11. What this notebook changed, and what it did not

**Changed.**

* The facet block is now a first-class part of `StackConfig`, not an
  afterthought: `views` supply content, `facets` supply style, and both are
  centred within the recovered group.
* The selection statistic is within-group η². Raw MI is confounded by the
  prompt and, for discrete facets, degenerate — it reports 0.0 for the person
  and opener families.
* The facet block helps **as a joint block**, not family by family: no single
  numeric family beats the control on the full stack, but voice + punctuation
  does (+0.027) and everything does (+0.043).
* Three families are retired on evidence: structure, emoji, citation.
* The in-style pool loader now filters to the 18 labels, drops train
  duplicates, and namespaces pool prompt ids — three silent failure modes that
  would otherwise corrupt the training set.

**Did not change.**

* The grouping: still recovered, still `k=3, cut=0.88`, still tuned at pool
  scale.
* The decision rule: still `argmax_k P(y=k|x)/N_k`, still with assignment
  inside recovered groups.
* The honest CV protocol: still `GroupKFold` on prompt ids, still the only
  number quoted as a forecast.

The next measurement is not another facet. It is the leaderboard: five
variants, five rows, one gap column.
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

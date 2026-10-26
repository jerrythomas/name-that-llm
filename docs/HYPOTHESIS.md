# Name That LLM — Hypothesis Register

> **Provenance.** This document records the project owner's brief of
> 2026-10-26, *before* any facet experiment had been run. Sections 1–5 restate
> the proposal in its own terms. Sections 6–8 add the constraints and the
> measurement protocol that turn it into falsifiable claims. Section 9 is the
> experiment register; section 10 is the running record of what actually
> worked. Update 9 and 10 after every submission — that is the point of the
> document.
>
> Nothing in sections 1–5 is a claim that a facet *works*. It is the list of
> things worth testing, written down so that later results can be attributed
> to a hypothesis instead of to memory.

---

## 1. The proposition in one paragraph

A passage's **content** is chosen by the prompt; the **way it is written** is
chosen by the model. If that is true, then enough independent stylistic
measurements — word choice, punctuation, sentence length, formatting,
hedging, person, citations, deviation from the prompt — should jointly
identify the author even when the prompt is held constant. Each measurement
is a *facet*. Facets may overlap or interact with the prompt's subject,
objective, and context, so the useful question is not "which facet wins" but
"which facet carries author signal *after* the prompt has been controlled
for, and which combinations of facets improve the decision".

The end goal is a classifier over **prompt + response + derived metadata**
that is enriched with as many quantitative metrics as can be defended, with
the winning combinations discovered empirically rather than assumed.

---

## 2. Facet inventory (as proposed)

The original checklist, with an honest note on whether each facet is
observable on the hidden test set. "Response" means the passage text alone;
"group" means response + the prompt groups recovered transductively.

| # | Facet | What it is meant to capture | Operationalisation | Observable at test? |
|---|-------|-----------------------------|--------------------|---------------------|
| F1 | Words and phrase usage | vocabulary habits, favoured constructions | word 1–2-gram TF-IDF (`word` view) | yes — response |
| F2 | Punctuation | comma/semicolon/dash/quote habits | per-100-word punctuation rates | yes — response |
| F3 | Sentence length | rhythm, mean and variance | sentence split → mean, SD | yes — response |
| F4 | Repeated words / sentences | verbal tics, loops | type–token ratio, hapax rate, repeated n-grams | yes — response |
| F5 | Structure | bullets, numbered lists, headings, tables, code | markdown probes | yes — response |
| F6 | Emojis | affect signalling | unicode-range count | yes — response |
| F7 | Bias | stance, framing, refusal, moral register | **not yet operationalised** — needs a reference frame | yes, but proxy only |
| F8 | Deviation from input prompt | drift, answering a different question | cosine to the recovered group centroid (prompt reconstruction) | yes — group proxy |
| F9 | Adherence to context / guardrails / objective | instruction-following, refusals, caveats | refusal and caveat lexicon; centroid distance | yes — group proxy |
| F10 | N-grams | local collocations | char 2–5-gram TF-IDF (`char` view) | yes — response |
| F11 | Citations, references | bracketed refs, author-year, URLs | regex probes | yes — response |
| F12 | Wording: first person, personification | voice, stance, agency | pronoun-rate families | yes — response |
| F13 | Does the response stay sane by topic? | consistency across subjects | per-model cross-prompt dispersion (train-time diagnostic) | train-time only |

Mapping to code: F1/F10 are the existing `word`/`char` views
(`src/ntl/views.py`); F2–F6, F11, F12 are response facets
(`src/ntl/facets.py`); F8/F9 are the `group` facet family; F7 and F13 are
open questions, not implemented.

---

## 3. Prompt categorisation (as proposed)

The second half of the brief: group the *prompts*, then ask whether a model's
facet profile shifts by prompt type. The proposed axes:

| # | Prompt axis | Why it might matter |
|---|-------------|---------------------|
| P1 | Domain | a model may have a distinctive register in philosophy vs code |
| P2 | Clarity | ambiguous prompts invite boilerplate; clear ones invite specificity |
| P3 | One-shot / multi-shot | exemplars change formatting and length |
| P4 | Number of questions | multi-part answers have different structure |
| P5 | Follow-up vs fresh question | follow-ups may be shorter, more conversational |
| P6 | Detail / depth requested | controls length and hedging |
| P7 | Guardrails / restrictions | safety prompts trigger refusals and caveats |
| P8 | *(added)* Prompt length | short prompts → longer, more elaborate answers |
| P9 | *(added)* Requested format | "bullet points", "table", "essay" directly sets structure |
| P10 | *(added)* Persona / role assignment | "you are a physicist" shifts vocabulary |
| P11 | *(added)* Language / register | formal vs casual instruction |
| P12 | *(added)* Safety-sensitive subject | refusals, disclaimers, hedging |
| P13 | *(added)* Math / code content | notation, code fences, precision markers |

**The hard constraint.** `test.csv` contains `id, text` and nothing else —
verified locally against `data/raw/test.csv`. There is no prompt text and no
`prompt_id`. Therefore every prompt axis P1–P13 is **not observable at
prediction time**. It can only enter the model in one of two indirect ways:

1. **Recovered pseudo-prompts.** Passages answering the same prompt cluster
   tightly in content-word TF-IDF (99.8% of nearest neighbours share a prompt
   on train). The group centroid reconstructs the prompt well enough to
   compute F8/F9, and a topic model over centroids can *infer* P1/P12 — but
   the inferred value is constant within a group, so it cannot separate the
   18 sibling passages.
2. **Training-time stratification.** Prompt categories can define separate
   training slices (a "philosophy" model, a "code" model). At test time the
   slice must be inferred from the recovered group, which is noisy but
   testable.

This is the single most important correction to the naive plan: **prompt-side
features cannot identify an author inside a prompt group.** They can only
modulate the prior or the choice of classifier. Any variant that claims
otherwise is measuring the prompt, not the author.

---

## 4. Proposed architecture

```
passage ──► facet extractor ──► per-facet score ──┐
   │                                              ├──► combiner ──► 18-way classifier
prompt ──► categoriser ──► group context ─────────┘
```

Three claims are bundled here and should be tested separately:

- **C1 (additivity).** Facets add information beyond n-gram views.
- **C2 (interaction).** The best facet weights depend on the prompt category.
- **C3 (stacking).** A learned combiner over facet scores beats a single
  classifier over concatenated features.

The planned mechanism for "score each facet then identify" is stacking: fit a
cheap per-facet model, feed its out-of-fold scores (plus the raw facets) into
a second classifier. The simpler alternative — standardise and concatenate —
is the baseline for C3.

---

## 5. Proposed experiment protocol (as specified)

1. **Enumerate** every derivable metric on the response, and on the
   prompt/response pair.
2. **EDA per facet**: show its distribution by model, and how it could
   influence classification.
3. **Select the top 5 combinations** from that evidence.
4. **One notebook per combination**, each self-contained and Kaggle-runnable,
   each writing `submission.parquet`.
5. **Run 5 submissions per day**, one per combination.
6. **Track** each combination: local cross-validation score, public
   leaderboard score, and the gap between them.
7. **Iterate**: keep the combinations whose local ranking predicts the public
   ranking; adjust weights and facet membership; repeat.

The tracker is not bookkeeping — it is the experiment's instrument. If local
CV does not rank combinations the same way the public LB does, then local CV
is not a usable selection signal and the whole loop has to change.

---

## 6. Reality check: what is measurable, and where the signal can live

| Signal | Available at fit | Available at predict | Can separate siblings in a group? |
|--------|------------------|----------------------|-----------------------------------|
| Response n-grams | yes | yes | **yes** — the core signal |
| Response facets F2–F6, F11, F12 | yes | yes | **yes** |
| Prompt text | yes (train has `prompt_id`, not text) | **no** | no |
| Prompt category P1–P13 | only if inferred from group | inferred | **no** — constant in group |
| Group-relative facets F8/F9 | yes (true groups) | yes (recovered groups) | **yes**, via rank/deviation |
| Test class mix | no | no | no (and rebalancing was measured harmful) |

Consequence: the facet programme must be judged on **within-group
discriminability**, not on raw mutual information. A facet can have huge MI
with the label purely because it is a proxy for the prompt — length is the
obvious trap, since some prompts ask for essays and others for one line.

The statistic that separates the two is **within-group eta²**: after
subtracting each prompt group's mean, how much of the remaining variance does
the model label explain? Values near zero mean "prompt proxy"; values above
the noise floor mean "author signal". It is the ranking device that will pick
the five combinations (`measure.within_group_eta2`, planned in
`src/ntl/measure.py`).

---

## 7. Falsifiable hypotheses

Each hypothesis has a prediction, a test, and a kill criterion. "Kill" means:
stop spending submissions on it.

### H1 — Response facets carry author signal
*Prediction:* at least one response facet family has within-group eta²
materially above the noise floor, and a facet-only classifier beats 1/18
under prompt-grouped CV.
*Test:* `measure.facet_signal` + `measure.facet_cv_score` per family.
*Kill:* every family's eta² is indistinguishable from zero **and** every
facet-only CV score is ≤ 1/18.

### H2 — Facets add to n-gram views
*Prediction:* `views + facets` beats `views` alone under grouped CV.
*Test:* variant v2/v3 vs v1, same folds.
*Kill:* no facet combination beats the 0.467 view-only stack by more than the
fold-noise floor (~0.04).

### H3 — Deviation from the reconstructed prompt is an author signal
*Prediction:* `g_centroid_cos` and `g_len_ratio` have non-zero within-group
eta²; the `group` family adds on top of response facets.
*Test:* eta² for the `group` family; v4 vs v3.
*Kill:* the group family is indistinguishable from a constant.

### H4 — Prompt category modulates facet usefulness
*Prediction:* per-domain facet profiles differ by model (an interaction), and
a domain-stratified classifier beats a single classifier at equal data.
*Test:* train-time only, using `prompt_id` + an inferred topic label; measure
the interaction term before committing a submission.
*Kill:* the interaction is within fold noise, or inferred domain purity is too
low to stratify at test time.

### H5 — A learned combiner beats concatenation
*Prediction:* stacking per-facet out-of-fold scores beats
standardise-and-concatenate.
*Test:* identical folds, identical base facets.
*Kill:* no improvement over concatenation.

### H6 — Local CV ranks combinations like the public LB
*Prediction:* Spearman correlation between local grouped-CV score and public
LB across the five variants is positive.
*Test:* the tracker, after the first week (≥ 25 submissions).
*Kill:* correlation ≤ 0 over a meaningful spread — then local CV is only a
sanity check, and the loop must select on LB alone.

---

## 8. Measurement protocol

- **Primary metric:** balanced accuracy, mean per-class recall over 18 models.
- **Split:** `GroupKFold` on `prompt_id` — the only split that matches the
  hidden test condition. Random CV is a different, harder task (measured
  0.077 vs 0.272 for the same model) and is never used for selection.
- **Grouping at validation:** recovered from text over the whole pool
  (`StackConfig.grouping_scope="pool"`), never from `prompt_id`.
- **Noise floor:** per-fold spread is ±0.04 at 2,699 passages. Quote the
  pooled score; treat differences below ~0.01 as noise, and below ~0.04 with
  suspicion unless repeated.
- **Within-group eta²:** the facet ranking statistic (section 6).
- **MI contrast:** raw mutual information is reported alongside eta² *only* to
  expose prompt proxies, never to select.
- **Local score is a ranking device, not an estimate.** It is computed on
  train-only by default so that all five variants are comparable and cheap;
  the submitted model may additionally train on the in-style pool, which the
  local score does not measure.
- **Tracker fields:** `variant, config_hash, views, facets, local_cv,
  local_folds, public_lb, n_train, n_pool, run_at, verdict, notes`.

---

## 9. Experiment register

The five combinations are **selected from the EDA in
`notebooks/03_facets.ipynb`**, then frozen into one notebook each. Fill the
last three columns after every submission.

| ID | Combination | Hypothesis | Local CV | Public LB | Δ | Verdict |
|----|-------------|-----------|----------|-----------|---|---------|
| v1 | `char + word + funcon` (control) | reference point for every other variant | TBD | TBD | | |
| v2 | v1 + `structure` + `punctuation` + `emoji` | formatting is an author habit (F2, F5, F6) | TBD | TBD | | |
| v3 | `funcon` + `length` + `lexical` + `person` + `hedging` | a content-free voice model (F3, F4, F12) | TBD | TBD | | |
| v4 | v1 + `group` facets | deviation from the reconstructed prompt matters (F8/F9) | TBD | TBD | | |
| v5 | v1 + all response + `group` facets | the full hypothesised signal set | TBD | TBD | | |

Already-measured ablations that constrain the search space:

| Configuration | Grouped CV | Conclusion |
|---------------|-----------|------------|
| char n-grams only | 0.309 | baseline |
| + word / function-word views | 0.365 | topic-free views help |
| + prompt centring | 0.425 | the prompt is removable |
| + Hungarian assignment in groups (the stack) | **0.467** | the current control |
| `punct` view in the union | 0.417 | hurts the union |
| `shape` view in the union | 0.442 | hurts the union |
| Sinkhorn to the observed test mix | ~0.10 under skew | do **not** rebalance |
| Two-stage lab → model cascade | 0.057 | rejected |
| Train on train + public in-style pool (lineage) | 0.563 → 0.604 | the largest known lever, independent of facets |

---

## 10. Running record — what worked, what did not

Append one line per experiment. Date, change, evidence, decision.

| Date | Change | Evidence | Decision |
|------|--------|----------|----------|
| 2026-10-2x | Grouped CV instead of random CV | 0.272 vs 0.077, same model | keep grouped CV for all selection |
| 2026-10-2x | Topic-free `funcon` view | +0.056 over char alone | keep |
| 2026-10-2x | Subtract the prompt group mean | +0.060 over best view | keep |
| 2026-10-2x | Hungarian assignment inside groups | +0.042 over independent argmax | keep |
| 2026-10-2x | Tune grouping at pool scale (`k=3, cut=0.88`) | ARI 0.967, purity 0.973 | keep, never tune on a fold |
| 2026-10-2x | Rebalance output to test mix | ~0.10 under skew | rejected |
| 2026-10-2x | Two-stage lab → model cascade | 0.057 | rejected |
| 2026-10-26 | Facet layer implemented (`src/ntl/facets.py`) | not yet measured | pending EDA |

---

## 11. Open questions

- **F7 (bias).** Without a reference frame, "bias" is not a well-defined
  scalar. Candidate proxies: refusal rate on a fixed probe set, stance
  lexicon density, moral-foundation word rates. None are implemented.
- **F13 (sanity by topic).** Measurable on train (same model, many prompts)
  but not on a single test passage. It could become a *prior* if the model is
  identified by other means, not a feature.
- **Domain inference quality.** How accurately can a recovered group centroid
  be assigned to a topic, and does the assignment survive the grouping error
  (ARI 0.967)?
- **Facet weighting.** Equal-weight concatenation is the default; the brief
  allows learned weights. Tune only after the tracker shows which variants
  survive, otherwise the weights fit noise.
- **Pool interaction.** Does the in-style pool change which facets matter? A
  facet that is rare in 2,699 passages may be common in 40k, and vice versa.
  Re-run the EDA on the pool once it is attached.

---

## 12. How this document is used (planned layout)

- `notebooks/03_facets.ipynb` produces the evidence for H1, H3, and the
  selection of v1–v5. It writes nothing to the leaderboard.
- `notebooks/04_v1_*.ipynb` … `04_v5_*.ipynb` are the five submission
  variants. Each prints a `TRACKER` JSON line and appends
  `/kaggle/working/tracker.csv`.
- `scripts/track.py` merges those rows into `metrics/tracker.csv`, where the
  public LB score is recorded after each submission.
- Section 9 is updated from the tracker, section 10 from the verdicts.

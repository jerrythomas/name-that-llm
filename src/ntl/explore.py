"""Exploratory analysis helpers.

Functions here return data (DataFrames / Series / dicts) rather than plotting,
so they compose cleanly in notebooks and stay headless-friendly. Charts live
in :mod:`ntl.plots`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LABELS, PROMPT_COL, PROVIDERS, TARGET_COL, TEXT_COL
from .features import FEATURE_COLS, add_text_features
from .groups import group_report, infer_groups
from .stylometry import (
    STOPWORDS,
    distinctive_profile,
    family_distinctive_words,
)
from .views import VIEWS, view_text


def label_distribution(train: pd.DataFrame) -> pd.DataFrame:
    """Per-model passage counts, share, and provider."""
    counts = train[TARGET_COL].value_counts().reindex(LABELS, fill_value=0)
    out = pd.DataFrame(
        {
            "model": counts.index,
            "n_passages": counts.values,
            "share": counts.values / counts.sum(),
            "provider": [PROVIDERS[m] for m in counts.index],
        }
    )
    return out.reset_index(drop=True)


def prompt_stats(train: pd.DataFrame) -> pd.DataFrame:
    """Passages and distinct models per prompt_id."""
    g = train.groupby(PROMPT_COL)
    out = pd.DataFrame(
        {
            "n_passages": g.size(),
            "n_models": g[TARGET_COL].nunique(),
        }
    )
    return out.sort_values("n_passages", ascending=False)


def per_model_features(train: pd.DataFrame) -> pd.DataFrame:
    """Stylistic features aggregated per model (mean)."""
    feats = add_text_features(train)
    return feats.groupby(TARGET_COL)[FEATURE_COLS].mean().reindex(LABELS)


def feature_zscores_per_model(train: pd.DataFrame) -> pd.DataFrame:
    """Per-model mean features, z-scored across models — heatmap-friendly."""
    means = per_model_features(train)
    return (means - means.mean()) / means.std(ddof=0).replace(0, 1)


def sample_passages(
    train: pd.DataFrame, model: str, n: int = 2, random_state: int = 0
) -> list[str]:
    """Random passages from a given model, for eyeballing its style."""
    subset = train[train[TARGET_COL] == model]
    return (
        subset[TEXT_COL].sample(n=min(n, len(subset)), random_state=random_state).tolist()
    )


def same_prompt_responses(
    train: pd.DataFrame, prompt_id: str | None = None, random_state: int = 0
) -> dict[str, str]:
    """One prompt answered by as many models as possible — compare styles
    side by side. Picks a random well-covered prompt if none given."""
    per_prompt_models = train.groupby(PROMPT_COL)[TARGET_COL].nunique()
    if prompt_id is None:
        candidates = per_prompt_models[per_prompt_models == per_prompt_models.max()]
        prompt_id = np.random.default_rng(random_state).choice(candidates.index)
    subset = train[train[PROMPT_COL] == prompt_id]
    return dict(zip(subset[TARGET_COL], subset[TEXT_COL], strict=False))


def distinctive_tokens(
    train: pd.DataFrame, top_n: int = 12, min_df: int = 5
) -> dict[str, pd.DataFrame]:
    """Most characteristic unigrams per model via one-vs-rest chi-squared.

    Returns ``{model: DataFrame[token, score]}`` — a quick read on each
    model's vocabulary tics (hedging words, formatting habits, etc.).
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_selection import chi2

    vec = CountVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        min_df=min_df,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z'][a-zA-Z'-]+\b",
    )
    X = vec.fit_transform(train[TEXT_COL].fillna(""))
    y = train[TARGET_COL]
    vocab = np.array(vec.get_feature_names_out())

    out: dict[str, pd.DataFrame] = {}
    for model in LABELS:
        binary = (y == model).astype(int).values
        scores, _ = chi2(X, binary)
        top = np.argsort(scores)[::-1][:top_n]
        out[model] = pd.DataFrame({"token": vocab[top], "score": scores[top]})
    return out


# ---------------------------------------------------------------------------
# Verbal fingerprints (log-odds z-scores)
# ---------------------------------------------------------------------------


def log_odds_fingerprints(
    train: pd.DataFrame,
    top_n: int = 12,
    min_count: int = 20,
    exclude: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Per-model log-odds fingerprint: the words a model actually reaches for.

    The z-score column says how strongly a word belongs to that model
    relative to the other seventeen — which is what separates a *habit* from
    a *topic*.
    """
    texts = train[TEXT_COL].to_numpy(dtype=object)
    labels = train[TARGET_COL].to_numpy(dtype=object)
    return distinctive_profile(
        texts,
        labels,
        LABELS,
        top_n=top_n,
        min_count=min_count,
        exclude=exclude if exclude is not None else set(STOPWORDS),
    )


def family_fingerprints(
    train: pd.DataFrame, top_n: int = 12, min_count: int = 20
) -> dict[str, pd.DataFrame]:
    """Per-lab log-odds fingerprint.

    Measured on the train set this is much stronger than the per-model
    fingerprint: labs have a register, individual siblings do not.
    """
    texts = train[TEXT_COL].to_numpy(dtype=object)
    labels = train[TARGET_COL].to_numpy(dtype=object)
    families = sorted(set(PROVIDERS.values()))
    return {
        family: family_distinctive_words(
            texts, labels, PROVIDERS, family, top_n=top_n, min_count=min_count
        )
        for family in families
    }


def repeated_phrases(
    train: pd.DataFrame, model: str, top_n: int = 15, n: int = 2
) -> pd.DataFrame:
    """The most repeated n-grams of a single model.

    This is the "does it have verbal tics?" question asked in the crudest
    possible way: which word sequences keep coming back?
    """
    from sklearn.feature_extraction.text import CountVectorizer

    subset = train[train[TARGET_COL] == model]
    vec = CountVectorizer(
        ngram_range=(n, n),
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]+\b",
    )
    counts = vec.fit_transform(subset[TEXT_COL].fillna(""))
    totals = np.asarray(counts.sum(axis=0)).ravel()
    vocab = vec.get_feature_names_out()
    order = np.argsort(totals)[::-1][:top_n]
    return pd.DataFrame(
        {"phrase": vocab[order], "count": totals[order].astype(int)}
    )


def sentence_openers(
    train: pd.DataFrame, model: str, top_n: int = 12, n_words: int = 3
) -> pd.DataFrame:
    """How a model *starts* its sentences.

    The opening of a sentence is one of the least topic-dependent habits a
    writer has: "It is worth noting that…", "There is a simple way to…",
    "Let me be clear:…".  Counting the first *n* words of every sentence
    turns that habit into a countable, comparable statistic.
    """
    import re as _re

    subset = train[train[TARGET_COL] == model]
    counter: dict[str, int] = {}
    for text in subset[TEXT_COL].fillna(""):
        for sentence in _re.split(r"(?<=[.!?])\s+", text):
            words = _re.findall(r"[A-Za-z][A-Za-z'-]+", sentence.lower())
            if len(words) < n_words:
                continue
            opener = " ".join(words[:n_words])
            counter[opener] = counter.get(opener, 0) + 1
    frame = pd.DataFrame(
        {"opener": list(counter), "count": list(counter.values())}
    ).sort_values("count", ascending=False)
    return frame.head(top_n).reset_index(drop=True)


def opening_fingerprint(
    train: pd.DataFrame, top_n: int = 10, n_words: int = 3
) -> dict[str, pd.DataFrame]:
    """Sentence openers per model, restricted to openers used at least twice."""
    out: dict[str, pd.DataFrame] = {}
    for model in LABELS:
        frame = sentence_openers(train, model, top_n=top_n, n_words=n_words)
        out[model] = frame[frame["count"] >= 2].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Recovered prompt groups
# ---------------------------------------------------------------------------


def recovered_group_report(train: pd.DataFrame, k: int = 3, cut: float = 0.95) -> dict:
    """How well does the inferred grouping recover the true prompt groups?"""
    texts = train[TEXT_COL].to_numpy(dtype=object)
    truth = train[PROMPT_COL].to_numpy(dtype=object)
    groups = infer_groups(texts, k=k, cut=cut)
    return group_report(groups, truth)


def view_examples(passage: str, max_chars: int = 240) -> dict[str, str]:
    """What each linguistic view does to one passage — the quickest way to
    see why one view is topic-free and another is not."""
    return {
        name: view_text(name, passage)[:max_chars]
        for name in ("char", "word", "shape", "funcon", "punct")
        if name in VIEWS
    }

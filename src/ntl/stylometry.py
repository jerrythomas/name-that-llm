"""The verbal fingerprint: which words a model actually repeats.

Motivation
----------
If two models can be told apart at all, it should be by *habit*: the words
they reach for, the hedges they use, the way they open a sentence.  This
module measures that directly, with the standard statistical test for
"distinctive words" — the Monroe et al. (2008) log-odds ratio with an
informative Dirichlet prior.

Why not a plain frequency table?  Raw counts are dominated by passage length
and by the topic of the prompt.  The log-odds z-score asks a sharper
question: *given that this word appears at all, how much more often does
model A use it than the other seventeen?*  The Dirichlet prior keeps rare
words from producing infinite z-scores, which is what makes the ranking
trustworthy on a corpus this small.

Two things this analysis revealed (see the notebook for the evidence):

* Family-level fingerprints are strong and legible — Google models reach for
  "profound / absolute / void", OpenAI models for "therefore / seem / turn".
* Sibling models *within* a family are **not** separable by vocabulary: their
  top words differ only in topic, which is prompt noise, not style.

That asymmetry is the whole reason the pipeline needs topic-free views.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")

#: Closed-class words are excluded from the "vocabulary" analyses: they
#: belong to the *style* views (see :mod:`ntl.views`), not to the lexical one.
STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "while",
        "although",
        "because",
        "as",
        "so",
        "than",
        "that",
        "which",
        "who",
        "whom",
        "whose",
        "this",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "theirs",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "we",
        "us",
        "our",
        "ours",
        "you",
        "your",
        "yours",
        "i",
        "me",
        "my",
        "mine",
        "in",
        "on",
        "at",
        "to",
        "from",
        "with",
        "without",
        "for",
        "of",
        "by",
        "over",
        "under",
        "between",
        "through",
        "into",
        "about",
        "against",
        "during",
        "before",
        "after",
        "above",
        "below",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "done",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "not",
        "no",
        "nor",
        "all",
        "any",
        "some",
        "each",
        "every",
        "both",
        "either",
        "neither",
        "other",
        "another",
        "such",
        "same",
        "own",
        "few",
        "little",
        "several",
        "much",
        "many",
        "more",
        "most",
        "less",
        "least",
        "also",
        "just",
        "only",
        "even",
        "still",
        "yet",
        "already",
        "always",
        "never",
        "often",
        "sometimes",
        "usually",
        "rarely",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "what",
        "whether",
        "because",
        "therefore",
        "thus",
        "hence",
        "consequently",
        "moreover",
        "furthermore",
        "however",
        "nevertheless",
        "nonetheless",
        "now",
        "then",
        "since",
        "until",
    ]
)


def tokenize(text: str, lower: bool = True) -> list[str]:
    """Split a passage into alphabetic word tokens."""
    words = _TOKEN_RE.findall(text)
    return [w.lower() for w in words] if lower else words


def log_odds_ratio(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    prior_a: np.ndarray | None = None,
    prior_b: np.ndarray | None = None,
    a0: float = 500.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Monroe et al. log-odds ratio and its z-score.

    Parameters
    ----------
    counts_a, counts_b:
        Raw token counts for the two corpora.
    prior_a, prior_b:
        Informative priors (usually the corpora's own token distributions).
        Defaults to uniform.
    a0:
        Strength of the prior in pseudo-tokens.

    Returns
    -------
    (delta, z):
        ``delta`` is the log-odds of the word appearing in corpus A rather
        than B; ``z`` is that effect size divided by its standard error and
        is the quantity worth ranking on.
    """
    counts_a = np.asarray(counts_a, dtype=float)
    counts_b = np.asarray(counts_b, dtype=float)
    n_a, n_b = counts_a.sum(), counts_b.sum()
    vocab = counts_a.size

    if prior_a is None:
        prior_a = np.full(vocab, 1.0 / vocab)
    if prior_b is None:
        prior_b = np.full(vocab, 1.0 / vocab)

    p_a = (counts_a + a0 * prior_a) / (n_a + a0)
    p_b = (counts_b + a0 * prior_b) / (n_b + a0)
    p_a = np.clip(p_a, 1e-12, 1 - 1e-12)
    p_b = np.clip(p_b, 1e-12, 1 - 1e-12)

    delta = np.log(p_a / p_b) - np.log((1 - p_a) / (1 - p_b))
    var = 1.0 / (n_a * p_a * (1 - p_a)) + 1.0 / (n_b * p_b * (1 - p_b))
    z = delta / np.sqrt(var)
    return delta, z


def _counts_by_label(
    texts: Sequence[str],
    labels: np.ndarray,
    inside_labels: set[str],
    stop: set[str],
) -> tuple[Counter, Counter]:
    """Document-frequency counts split into "inside" and "outside" corpora."""
    inside: Counter = Counter()
    outside: Counter = Counter()
    for text, label in zip(texts, labels, strict=False):
        toks = {t for t in tokenize(text) if t not in stop}
        (inside if label in inside_labels else outside).update(toks)
    return inside, outside


def _rank(
    inside: Counter,
    outside: Counter,
    top_n: int,
    min_count: int,
) -> pd.DataFrame:
    vocab = sorted(w for w in (set(inside) | set(outside)) if inside[w] + outside[w] >= min_count)
    if not vocab:
        return pd.DataFrame(columns=["word", "log_odds", "z"])
    a = np.array([inside[w] for w in vocab], dtype=float)
    b = np.array([outside[w] for w in vocab], dtype=float)
    prior_a = (a + 1.0) / (a.sum() + len(a))
    prior_b = (b + 1.0) / (b.sum() + len(b))
    delta, z = log_odds_ratio(a, b, prior_a, prior_b)
    order = np.argsort(z)[::-1][:top_n]
    return pd.DataFrame(
        {
            "word": [vocab[i] for i in order],
            "log_odds": np.round(delta[order], 2),
            "z": np.round(z[order], 1),
        }
    )


def distinctive_words(
    texts: Sequence[str],
    labels: np.ndarray,
    target: str,
    top_n: int = 12,
    min_count: int = 20,
    exclude: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Most characteristic words of *target* versus every other model.

    Returns a DataFrame with columns ``word``, ``log_odds`` and ``z``.
    """
    stop = set(exclude) if exclude is not None else set(STOPWORDS)
    inside, outside = _counts_by_label(texts, labels, {target}, stop)
    return _rank(inside, outside, top_n, min_count)


def distinctive_profile(
    texts: Sequence[str],
    labels: np.ndarray,
    targets: Sequence[str],
    top_n: int = 12,
    min_count: int = 20,
    exclude: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run :func:`distinctive_words` for several targets at once."""
    return {
        target: distinctive_words(
            texts, labels, target, top_n=top_n, min_count=min_count, exclude=exclude
        )
        for target in targets
    }


def family_distinctive_words(
    texts: Sequence[str],
    labels: np.ndarray,
    families: dict[str, str],
    family: str,
    top_n: int = 12,
    min_count: int = 20,
    exclude: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Words used more by every model of one lab than by any other lab.

    A family profile is the readable summary of "which lab is this" — and,
    measured on the train set, it is a far stronger signal than any
    individual model's profile.
    """
    stop = set(exclude) if exclude is not None else set(STOPWORDS)
    members = {m for m, f in families.items() if f == family}
    inside, outside = _counts_by_label(texts, labels, members, stop)
    return _rank(inside, outside, top_n, min_count)


def top_content_words(
    texts: Sequence[str],
    labels: np.ndarray,
    target: str,
    top_n: int = 15,
    exclude: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Most repeated content words of a model (an absolute view, not relative)."""
    stop = set(exclude) if exclude is not None else set(STOPWORDS)
    counter: Counter = Counter()
    for text, label in zip(texts, labels, strict=False):
        if label == target:
            counter.update(set(tokenize(text)) - stop)
    frame = pd.DataFrame({"word": list(counter), "count": list(counter.values())}).sort_values(
        "count", ascending=False
    )
    return frame.head(top_n).reset_index(drop=True)

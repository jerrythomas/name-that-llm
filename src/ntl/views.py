"""Linguistic *views*: different reductions of a passage into a feature space.

Why views at all?  Every passage answers a prompt, so a large part of its
surface form is chosen by the *prompt*, not by the *model*.  Vocabulary is
therefore the worst possible space to identify an author in: two models
answering the same prompt will agree on the nouns.  What differs between
sibling models of the same lab is *how* the passage is put together.

Each view below removes a different layer of topical content and leaves a
different layer of style:

============  =========================================================
view          what survives
============  =========================================================
``char``      everything: characters, words, punctuation (lexical + format)
``word``      word n-grams: vocabulary and word order
``shape``     word *shape* only (``X``/``x``): capitalisation habits
``funcon``    syntax only: function words kept, content words masked
``punct``     formatting only: punctuation and whitespace
============  =========================================================

``funcon`` and ``shape`` are the interesting ones, because they are
*invariant to topic*: two passages answering different prompts with the same
rhetorical structure look the same under them.  That is exactly what we
want when the prompt is a nuisance variable to be removed.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Closed-class words.  These carry grammar (who did what to whom, how certain,
# how connected), not topic.  A list rather than a library stop-word list, so
# that *content* words like "nothing" or "never" stay visible where they are
# meaningful as hedges.
FUNCTION_WORDS: frozenset[str] = frozenset(
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
        "ought",
        "need",
        "dare",
        "not",
        "no",
        "nor",
        "none",
        "never",
        "neither",
        "none",
        "nothing",
        "nowhere",
        "anyone",
        "anything",
        "everyone",
        "everything",
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
        "very",
        "too",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "what",
        "whether",
        "which",
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
        "here",
        "now",
        "then",
        "since",
        "until",
        "while",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "within",
        "without",
        "upon",
        "toward",
        "towards",
        "across",
        "around",
        "of",
        "off",
        "out",
        "up",
        "down",
        "in",
        "on",
        "at",
    ]
)

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_WORD_RE = re.compile(r"[^\W\d_]+")


def shape_transform(text: str) -> str:
    """Replace every alphabetic word by ``X`` (if capitalised) or ``x``.

    Keeps capitalisation, word length and punctuation; throws away identity.
    """
    return _WORD_RE.sub(lambda m: "X" if m.group(0)[0].isupper() else "x", text)


def function_transform(text: str) -> str:
    """Keep function words and punctuation, mask every content word.

    The result is a *syntactic skeleton* of the passage: the same rhetorical
    move ("however, ... therefore, ... it is ...") survives regardless of what
    the passage is about.
    """
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text):
        if not tok.isalnum():
            out.append(tok)
            continue
        low = tok.lower()
        out.append(low if low in FUNCTION_WORDS else "CONTENT")
    return " ".join(out)


def punct_transform(text: str) -> str:
    """Strip letters and digits; keep punctuation and whitespace only.

    This is the purest formatting view: dashes, commas, quotes, line breaks.
    """
    return re.sub(r"[A-Za-z0-9]+", " ", text)


@dataclass(frozen=True)
class ViewSpec:
    """A named text transform plus the TF-IDF settings used on top of it."""

    name: str
    transform: Callable[[str], str]
    vectorizer_kwargs: dict

    def apply(self, text: str) -> str:
        return self.transform(text)


VIEWS: dict[str, ViewSpec] = {
    "char": ViewSpec(
        "char",
        lambda t: t,
        dict(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=3,
            sublinear_tf=True,
            lowercase=False,
            max_features=300_000,
        ),
    ),
    "word": ViewSpec(
        "word",
        lambda t: t,
        dict(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            lowercase=True,
            max_features=200_000,
            token_pattern=r"(?u)\b\w+\b|[^\w\s]",
        ),
    ),
    "shape": ViewSpec(
        "shape",
        shape_transform,
        dict(
            analyzer="char",
            ngram_range=(2, 6),
            min_df=3,
            sublinear_tf=True,
            lowercase=False,
            max_features=100_000,
        ),
    ),
    "funcon": ViewSpec(
        "funcon",
        function_transform,
        dict(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=3,
            sublinear_tf=True,
            lowercase=True,
            max_features=100_000,
            token_pattern=r"(?u)\b\w+\b|[^\w\s]",
        ),
    ),
    "punct": ViewSpec(
        "punct",
        punct_transform,
        dict(
            analyzer="char",
            ngram_range=(1, 4),
            min_df=3,
            sublinear_tf=True,
            lowercase=False,
            max_features=50_000,
        ),
    ),
}

DEFAULT_VIEWS: tuple[str, ...] = ("char", "word", "funcon")


def view_text(view: str, text: str) -> str:
    """Apply the transform of a named view to one text."""
    return VIEWS[view].apply(text)


def fit_views(
    texts: Sequence[str], views: Iterable[str] = DEFAULT_VIEWS
) -> dict[str, TfidfVectorizer]:
    """Fit one TF-IDF vectorizer per view on *texts*."""
    fitted: dict[str, TfidfVectorizer] = {}
    for name in views:
        spec = VIEWS[name]
        vec = TfidfVectorizer(dtype=np.float32, **spec.vectorizer_kwargs)
        vec.fit([spec.apply(t) for t in texts])
        fitted[name] = vec
    return fitted


def transform_views(
    fitted: dict[str, TfidfVectorizer],
    texts: Sequence[str],
    views: Iterable[str] = DEFAULT_VIEWS,
    normalize_rows: bool = True,
) -> sparse.csr_matrix:
    """Vectorise *texts* under every view and stack the blocks.

    Each block is L2-normalised so that no view dominates the shared weight
    vector purely because it has more features.
    """
    blocks = []
    for name in views:
        spec = VIEWS[name]
        block = fitted[name].transform([spec.apply(t) for t in texts])
        blocks.append(normalize(block) if normalize_rows else block)
    return sparse.hstack(blocks, format="csr")


class ViewUnion(BaseEstimator, TransformerMixin):
    """scikit-learn transformer exposing several views as one feature space.

    Lets a view stack be dropped straight into a ``Pipeline``.
    """

    def __init__(self, views: Sequence[str] | None = None):
        self.views = list(views) if views is not None else list(DEFAULT_VIEWS)

    def fit(self, X, y=None):  # noqa: N803 - sklearn convention
        self.vecs_ = fit_views(X, self.views)
        return self

    def transform(self, X):  # noqa: N803 - sklearn convention
        return transform_views(self.vecs_, X, self.views)

    def fit_transform(self, X, y=None, **fit_params):  # noqa: N803
        self.vecs_ = fit_views(X, self.views)
        return transform_views(self.vecs_, X, self.views)

    def get_feature_names_out(self, input_features=None):
        names: list[str] = []
        for name in self.views:
            names.extend(f"{name}__{f}" for f in self.vecs_[name].get_feature_names_out())
        return np.array(names, dtype=object)

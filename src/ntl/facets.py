"""Facets — the checklist of style questions, turned into numbers.

The competition gives the response text and nothing else: ``test.csv`` has
``id, text``.  So a facet is *usable* only if it can be computed from the
response alone, or from the response plus the prompt groups this package
recovers transductively.  That one constraint decides the whole design:

======================  ===========================  ==========================
question                computable at test time?      how
======================  ===========================  ==========================
words and phrases       yes                          n-grams (:mod:`ntl.views`)
punctuation             yes                          per-100-word rates
sentence length         yes                          sentence split
repeated words          yes                          type-token ratio, hapax
structure (bullets ...) yes                          markdown probes
emojis                  yes                          unicode ranges
wording (person)        yes                          pronoun rates
hedging / boosters      yes                          closed word lists
citations               yes                          bracket / year probes
response length         yes                          counts
deviation from prompt   *proxy*                      cosine to the recovered
                                                      group centroid
prompt domain           *proxy*                      centroid topic — but it is
                                                      constant inside a group
prompt clarity / shots  **no**                       the prompt is not in test
prompt-response distance**no**                       no prompt text at test
======================  ===========================  ==========================

Two consequences follow, and they explain the two halves of this module:

* :func:`response_facets` is the honest core: every column is a function of
  the passage alone, so it transfers to the hidden test unchanged.
* :func:`group_facets` is the transductive bonus: it compares a passage with
  the *recovered* prompt group it belongs to.  It cannot separate passages
  whose groups are already merged, but it does capture "this answer is
  unusually long for this question", which raw rates cannot.

Prompt-side facets (domain, clarity, few-shot, guardrails) are deliberately
absent because they are not observable at prediction time.  They can still
shape *training* (domain-stratified models), but :mod:`ntl.measure` shows the
inferred domain is constant within a recovered group, so it carries no
within-group signal and cannot identify an author.

Naming: ``f_`` columns are response facets, ``g_`` columns are group facets.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Lexical resources
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_EMOJI_RE = re.compile(
    r"[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff"
    r"\U00002190-\U000021ff\U00002b00-\U00002bff]"
)

FIRST_SINGULAR = frozenset({"i", "me", "my", "mine", "myself"})
FIRST_PLURAL = frozenset({"we", "us", "our", "ours", "ourselves"})
SECOND_PERSON = frozenset({"you", "your", "yours", "yourself", "yourselves"})
THIRD_PERSON = frozenset(
    {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"}
)

HEDGES = frozenset(
    {
        "perhaps",
        "maybe",
        "possibly",
        "probably",
        "likely",
        "arguably",
        "seemingly",
        "apparently",
        "roughly",
        "approximately",
        "somewhat",
        "fairly",
        "relatively",
        "generally",
        "typically",
        "often",
        "sometimes",
        "tends",
        "seems",
        "appears",
        "suggests",
        "may",
        "might",
        "could",
    }
)
BOOSTERS = frozenset(
    {
        "clearly",
        "certainly",
        "definitely",
        "indeed",
        "obviously",
        "undoubtedly",
        "absolutely",
        "always",
        "never",
        "must",
        "crucially",
        "importantly",
        "notably",
        "especially",
        "particularly",
    }
)
CONNECTIVES = frozenset(
    {
        "however",
        "therefore",
        "moreover",
        "furthermore",
        "nevertheless",
        "nonetheless",
        "consequently",
        "thus",
        "hence",
        "additionally",
        "meanwhile",
        "similarly",
        "conversely",
        "specifically",
        "ultimately",
        "overall",
        "finally",
        "first",
        "second",
        "third",
    }
)

_CONTRACTION_RE = re.compile(r"\b(?:[A-Za-z]+)'(?:s|t|re|ve|ll|d|m)\b", re.IGNORECASE)
_CITE_BRACKET_RE = re.compile(r"\[\d{1,3}\]")
_CITE_YEAR_RE = re.compile(r"\((?:[A-Z][A-Za-z.\-]+(?: et al\.)?,\s*)?\d{4}[a-z]?\)")
_URL_RE = re.compile(r"https?://\S+")
_ALLCAPS_RE = re.compile(r"\b[A-Z]{2,}\b")

_MD_HEADER_RE = re.compile(r"(?m)^#{1,6}\s")
_MD_BULLET_RE = re.compile(r"(?m)^\s*[-*\u2022]\s")
_MD_NUMBERED_RE = re.compile(r"(?m)^\s*\d+[.)]\s")
_MD_TABLE_RE = re.compile(r"(?m)^\s*\|.*\|\s*$")
_MD_BOLD_RE = re.compile(r"\*\*|__")
_MD_CODE_RE = re.compile(r"```|`[^`\n]+`")
_MD_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")

_PUNCTUATION: dict[str, str] = {
    "comma": ",",
    "semicolon": ";",
    "colon": ":",
    "emdash": "\u2014",
    "endash": "\u2013",
    "paren": "()",
    "quote": "\"'",
    "exclam": "!",
    "question": "?",
    "ellipsis": "...",
    "slash": "/",
    "percent": "%",
}

# ---------------------------------------------------------------------------
# Feature families — the vocabulary the notebooks and the tracker speak
# ---------------------------------------------------------------------------

LENGTH_FACETS = (
    "f_words",
    "f_chars",
    "f_sentences",
    "f_paragraphs",
    "f_words_per_sentence",
    "f_sentence_len_sd",
    "f_avg_word_len",
    "f_long_word_rate",
    "f_short_word_rate",
)
LEXICAL_FACETS = ("f_ttr", "f_hapax_rate", "f_stopword_rate", "f_contraction_rate")
PUNCTUATION_FACETS = tuple(f"f_punct_{name}" for name in _PUNCTUATION)
STRUCTURE_FACETS = (
    "f_md_header",
    "f_md_bullet",
    "f_md_numbered",
    "f_md_table",
    "f_md_code",
    "f_md_bold",
    "f_md_link",
    "f_newline_rate",
    "f_digits_rate",
    "f_allcaps_rate",
)
EMOJI_FACETS = ("f_emoji_rate",)
PERSON_FACETS = (
    "f_first_singular",
    "f_first_plural",
    "f_second_person",
    "f_third_person",
    "f_impersonal_one",
)
HEDGING_FACETS = ("f_hedge_rate", "f_booster_rate", "f_connective_rate")
CITATION_FACETS = ("f_cite_bracket", "f_cite_year", "f_url")
OPENER_FACETS = (
    "f_open_it",
    "f_open_there",
    "f_open_the",
    "f_open_wh",
    "f_question_sentence_rate",
)
GROUP_FACETS = (
    "g_word_rank",
    "g_word_z",
    "g_sentence_z",
    "g_ttr_z",
    "g_comma_z",
    "g_len_ratio",
    "g_centroid_cos",
    "g_size_log",
)

FAMILIES: dict[str, tuple[str, ...]] = {
    "length": LENGTH_FACETS,
    "lexical": LEXICAL_FACETS,
    "punctuation": PUNCTUATION_FACETS,
    "structure": STRUCTURE_FACETS,
    "emoji": EMOJI_FACETS,
    "person": PERSON_FACETS,
    "hedging": HEDGING_FACETS,
    "citation": CITATION_FACETS,
    "opener": OPENER_FACETS,
    "group": GROUP_FACETS,
}

RESPONSE_FAMILIES: tuple[str, ...] = tuple(f for f in FAMILIES if f != "group")
ALL_RESPONSE_FACETS: tuple[str, ...] = tuple(
    column for family in RESPONSE_FAMILIES for column in FAMILIES[family]
)
ALL_FACETS: tuple[str, ...] = tuple(dict.fromkeys(ALL_RESPONSE_FACETS + GROUP_FACETS))

_FAMILY_OF: dict[str, str] = {
    column: family for family, cols in FAMILIES.items() for column in cols
}


def family_of(column: str) -> str:
    """Which family a facet column belongs to (``"unknown"`` if unregistered)."""
    return _FAMILY_OF.get(column, "unknown")


def facet_columns(families: Iterable[str] | None = None) -> list[str]:
    """Resolve family names *and/or* column names into an ordered column list.

    ``facet_columns(["structure", "g_centroid_cos"])`` is valid and returns
    the structure family followed by that one group facet.  Unknown names
    raise, so a typo in a notebook fails loudly instead of silently training
    on nothing.
    """
    if families is None:
        return list(ALL_FACETS)
    out: list[str] = []
    for name in families:
        if name in FAMILIES:
            out.extend(FAMILIES[name])
        elif name in ALL_FACETS:
            out.append(name)
        else:
            raise KeyError(f"unknown facet family/column {name!r}")
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Response facets
# ---------------------------------------------------------------------------


def _rate(count: float, n_words: int) -> float:
    return 100.0 * float(count) / max(n_words, 1)


def _analyse(text: str) -> dict[str, float]:
    """Every response-side facet for one passage."""
    text = "" if text is None else str(text)
    words = _WORD_RE.findall(text)
    lower = [w.lower() for w in words]
    n_words = len(words)
    n_chars = len(text)

    sentences = [s for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    n_sentences = len(sentences)
    sent_lens = [len(_WORD_RE.findall(s)) for s in sentences] or [0]
    paragraphs = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]

    first_100 = lower[:100]
    ttr = len(set(first_100)) / len(first_100) if first_100 else 0.0
    counts: dict[str, int] = {}
    for word in lower:
        counts[word] = counts.get(word, 0) + 1
    hapax = sum(1 for c in counts.values() if c == 1)
    stop = sum(1 for w in lower if w in ENGLISH_STOP_WORDS)

    openers = [_WORD_RE.findall(s)[:1] for s in sentences]
    openers = [o[0].lower() for o in openers if o]
    n_openers = max(len(openers), 1)

    row: dict[str, float] = {
        "f_words": float(n_words),
        "f_chars": float(n_chars),
        "f_sentences": float(n_sentences),
        "f_paragraphs": float(len(paragraphs)),
        "f_words_per_sentence": n_words / max(n_sentences, 1),
        "f_sentence_len_sd": float(np.std(sent_lens)) if len(sent_lens) > 1 else 0.0,
        "f_avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "f_long_word_rate": _rate(sum(1 for w in words if len(w) >= 8), n_words),
        "f_short_word_rate": _rate(sum(1 for w in words if len(w) <= 3), n_words),
        "f_ttr": ttr,
        "f_hapax_rate": hapax / max(n_words, 1),
        "f_stopword_rate": stop / max(n_words, 1),
        "f_contraction_rate": _rate(len(_CONTRACTION_RE.findall(text)), n_words),
        "f_md_header": float(len(_MD_HEADER_RE.findall(text))),
        "f_md_bullet": float(len(_MD_BULLET_RE.findall(text))),
        "f_md_numbered": float(len(_MD_NUMBERED_RE.findall(text))),
        "f_md_table": float(len(_MD_TABLE_RE.findall(text))),
        "f_md_code": float(len(_MD_CODE_RE.findall(text))),
        "f_md_bold": float(len(_MD_BOLD_RE.findall(text))),
        "f_md_link": float(len(_MD_LINK_RE.findall(text))),
        "f_newline_rate": _rate(text.count("\n"), n_words),
        "f_digits_rate": _rate(sum(ch.isdigit() for ch in text), n_words),
        "f_allcaps_rate": _rate(len(_ALLCAPS_RE.findall(text)), n_words),
        "f_emoji_rate": _rate(len(_EMOJI_RE.findall(text)), n_words),
        "f_first_singular": _rate(sum(w in FIRST_SINGULAR for w in lower), n_words),
        "f_first_plural": _rate(sum(w in FIRST_PLURAL for w in lower), n_words),
        "f_second_person": _rate(sum(w in SECOND_PERSON for w in lower), n_words),
        "f_third_person": _rate(sum(w in THIRD_PERSON for w in lower), n_words),
        "f_impersonal_one": _rate(sum(w == "one" for w in lower), n_words),
        "f_hedge_rate": _rate(sum(w in HEDGES for w in lower), n_words),
        "f_booster_rate": _rate(sum(w in BOOSTERS for w in lower), n_words),
        "f_connective_rate": _rate(sum(w in CONNECTIVES for w in lower), n_words),
        "f_cite_bracket": _rate(len(_CITE_BRACKET_RE.findall(text)), n_words),
        "f_cite_year": _rate(len(_CITE_YEAR_RE.findall(text)), n_words),
        "f_url": _rate(len(_URL_RE.findall(text)), n_words),
        "f_open_it": sum(o == "it" for o in openers) / n_openers,
        "f_open_there": sum(o == "there" for o in openers) / n_openers,
        "f_open_the": sum(o == "the" for o in openers) / n_openers,
        "f_open_wh": sum(o in {"what", "why", "how", "when", "where", "who"} for o in openers)
        / n_openers,
        "f_question_sentence_rate": sum(s.rstrip().endswith("?") for s in sentences)
        / max(n_sentences, 1),
    }
    for name, chars in _PUNCTUATION.items():
        row[f"f_punct_{name}"] = _rate(sum(text.count(ch) for ch in chars), n_words)
    return row


def response_facets(texts: Sequence[str]) -> pd.DataFrame:
    """One row of response-only facets per passage.

    Every column is computable from the hidden test set, because the hidden
    test set contains exactly this input: the passage text.
    """
    rows = [_analyse(text) for text in texts]
    frame = pd.DataFrame(rows)
    return frame.reindex(columns=list(ALL_RESPONSE_FACETS), fill_value=0.0).astype(float)


# ---------------------------------------------------------------------------
# Group facets (transductive)
# ---------------------------------------------------------------------------


def _group_rank(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.full(len(values), 0.5)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        if len(idx) < 2:
            continue
        order = np.argsort(values[idx], kind="stable")
        ranks = np.empty(len(idx))
        ranks[order] = np.arange(len(idx)) / (len(idx) - 1)
        out[idx] = ranks
    return out


def _group_z(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values))
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        if len(idx) < 2:
            continue
        block = values[idx]
        sd = float(block.std())
        out[idx] = 0.0 if sd < 1e-12 else (block - block.mean()) / sd
    return out


def _centroid_similarity(texts: Sequence[str], groups: np.ndarray) -> np.ndarray:
    """Leave-one-out cosine between a passage and its group's content centroid.

    This is the computable stand-in for "distance from the prompt": the group
    centroid is what all answers to one prompt share, i.e. a reconstruction of
    the prompt itself.  A passage close to the centroid answered the question
    on topic; a far one drifted.  Self is removed so that a two-member group
    cannot look artificially typical.
    """
    n = len(texts)
    if n < 2:
        return np.ones(n)
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, dtype=np.float32)
    X = normalize(vec.fit_transform(texts))
    _, inverse = np.unique(groups, return_inverse=True)
    membership = sparse.csr_matrix(
        (np.ones(n), (np.arange(n), inverse)), shape=(n, int(inverse.max()) + 1)
    )
    sums = membership.T @ X  # (n_groups, vocab)
    row_sums = sums[inverse]  # (n, vocab)
    dot_all = np.asarray(X.multiply(row_sums).sum(axis=1)).ravel()
    norm_sq = np.asarray(sums.multiply(sums).sum(axis=1)).ravel()[inverse]
    loo_dot = dot_all - 1.0  # x . (S - x) = x.S - 1
    loo_norm = np.sqrt(np.maximum(norm_sq - 2.0 * dot_all + 1.0, 0.0))
    cos = np.divide(loo_dot, loo_norm, out=np.zeros_like(loo_dot), where=loo_norm > 1e-9)
    return np.clip(cos, -1.0, 1.0)


def group_facets(
    texts: Sequence[str],
    groups: Sequence,
    base: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Facets that describe a passage *relative to its recovered prompt group*.

    ``groups`` may be the true ``prompt_id`` (at training time) or the groups
    recovered by :func:`ntl.groups.infer_groups` (at prediction time).  The
    features are defined so that they mean the same thing either way.
    """
    groups = np.asarray(groups)
    base = response_facets(texts) if base is None else base
    words = base["f_words"].to_numpy(dtype=float)
    sentences = base["f_sentences"].to_numpy(dtype=float)
    ttr = base["f_ttr"].to_numpy(dtype=float)
    commas = base["f_punct_comma"].to_numpy(dtype=float)

    group_median = pd.Series(words).groupby(groups).transform("median").to_numpy(dtype=float)
    sizes = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy(dtype=float)

    out = pd.DataFrame(index=base.index)
    out["g_word_rank"] = _group_rank(words, groups)
    out["g_word_z"] = _group_z(np.log1p(words), groups)
    out["g_sentence_z"] = _group_z(np.log1p(sentences), groups)
    out["g_ttr_z"] = _group_z(ttr, groups)
    out["g_comma_z"] = _group_z(commas, groups)
    out["g_len_ratio"] = words / np.maximum(group_median, 1.0)
    out["g_centroid_cos"] = _centroid_similarity(texts, groups)
    out["g_size_log"] = np.log1p(sizes)
    return out[list(GROUP_FACETS)].astype(float)


def facet_frame(
    texts: Sequence[str],
    groups: Sequence | None = None,
    families: Iterable[str] | None = None,
) -> pd.DataFrame:
    """The full facet frame for *texts*, restricted to *families*.

    With ``groups=None`` only response facets are returned; asking for the
    ``group`` family without groups is an error, because those columns would
    silently be meaningless.
    """
    columns = facet_columns(families)
    needs_group = any(c in GROUP_FACETS for c in columns)
    if needs_group and groups is None:
        raise ValueError("group facets require groups")
    base = response_facets(texts)
    if needs_group:
        base = pd.concat([base, group_facets(texts, groups, base)], axis=1)
    return base.reindex(columns=columns, fill_value=0.0).astype(float)

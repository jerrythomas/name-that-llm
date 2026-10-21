"""Lightweight stylistic features for LLM attribution.

The competition blurb hints the signal lives in *style*: word choice,
punctuation, sentence structure — not topic or length shortcuts. These are
cheap, model-agnostic features we can explore (and later feed to classical
classifiers) before reaching for embeddings.
"""

from __future__ import annotations

import re

import pandas as pd

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"[.!?]+")

# Punctuation probes: each lab's post-training tends to leave fingerprints here.
_PUNCT = {
    "comma": ",",
    "period": ".",
    "exclamation": "!",
    "question": "?",
    "semicolon": ";",
    "colon": ":",
    "dash": "—",  # em dash
    "hyphen": "-",
    "paren": "()",
    "quote": "\"'",
    "ellipsis": "...",
    "asterisk": "*",
    "hash": "#",
}


def add_text_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Return a copy of *df* with stylistic feature columns appended.

    Features cover length, lexical richness, sentence shape, punctuation
    habits and markdown artifacts (headers, bullets, bold).
    """
    out = df.copy()
    texts = out[text_col].fillna("")

    out["n_chars"] = texts.str.len()
    out["n_letters"] = texts.str.count(r"[A-Za-z]")
    out["n_digits"] = texts.str.count(r"\d")
    out["n_upper"] = texts.str.count(r"[A-Z]")
    out["n_whitespace"] = texts.str.count(r"\s")
    out["n_newline"] = texts.str.count(r"\n")

    words = texts.str.findall(_WORD_RE)
    out["n_words"] = words.apply(len)
    out["avg_word_len"] = words.apply(
        lambda ws: sum(len(w) for w in ws) / len(ws) if ws else 0.0
    )
    out["lexical_diversity"] = words.apply(
        lambda ws: len(set(w.lower() for w in ws)) / len(ws) if ws else 0.0
    )

    sents = texts.str.findall(_SENT_RE)
    out["n_sentences"] = sents.apply(len)
    out["avg_sent_len"] = out["n_words"] / out["n_sentences"].clip(lower=1)

    for name, char in _PUNCT.items():
        out[f"punct_{name}"] = texts.str.count(re.escape(char)) / out["n_words"].clip(
            lower=1
        )

    # Markdown / structure artifacts (only "light cleanup" was applied per the
    # data description, so in-passage markdown likely survives).
    out["md_header"] = texts.str.count(r"(?m)^#{1,6}\s")
    out["md_bullet"] = texts.str.count(r"(?m)^\s*([-*•]|\d+\.)\s")
    out["md_bold"] = texts.str.count(r"\*\*|__")
    out["md_code"] = texts.str.count(r"`")
    out["md_math"] = texts.str.count(r"\\\[|\\\(|\$")

    return out


FEATURE_COLS: list[str] = [
    "n_chars",
    "n_letters",
    "n_digits",
    "n_upper",
    "n_newline",
    "n_words",
    "avg_word_len",
    "lexical_diversity",
    "n_sentences",
    "avg_sent_len",
    *[f"punct_{name}" for name in _PUNCT],
    "md_header",
    "md_bullet",
    "md_bold",
    "md_code",
    "md_math",
]

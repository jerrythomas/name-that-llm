"""Classical baselines for authorship attribution.

The metric is **balanced accuracy** (mean per-class recall), so we evaluate
with that and use ``class_weight='balanced'``. Two complementary views:

* word n-grams  — vocabulary / phrasing fingerprints
* char n-grams  — punctuation, spacing, formatting fingerprints (strong for
  authorship attribution, and robust to the unseen-prompt shift)

A word of warning about the *evaluation*: random cross-validation does not
flatter this problem, it makes it a *different and much harder* task.  Every
prompt is answered by all 18 models, so a random split puts ~13.6 of each
validation passage's siblings into the training fold — each carrying a
*different* label.  The classifier is then asked to map one piece of
content to 18 different authors, and the cheapest way to fit that is to
ignore the content — and with it, the style.  The score collapses to 0.077
for words and 0.116 for characters, against 0.272 and 0.309 under
prompt-grouped folds.  Use :func:`grouped_cv_scores`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from .config import PROMPT_COL, RANDOM_STATE, TARGET_COL, TEXT_COL


def _as_text_array(series: pd.Series) -> np.ndarray:
    """Return a plain numpy object array of strings.

    pandas 3.x defaults string columns to a pyarrow-backed dtype whose
    ``.values`` is not fancy-indexable by sklearn's parallel CV workers.
    """
    return series.fillna("").astype(object).to_numpy(dtype=object)


def build_word_pipeline(ngram_range: tuple[int, int] = (1, 2), min_df: int = 2) -> Pipeline:
    """TF-IDF word n-grams + logistic regression."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=ngram_range,
                    min_df=min_df,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0),
            ),
        ]
    )


def build_char_pipeline(
    ngram_range: tuple[int, int] = (2, 5),
    min_df: int = 3,
    max_features: int = 300_000,
) -> Pipeline:
    """TF-IDF char n-grams + logistic regression."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_features=max_features,
                    sublinear_tf=True,
                    lowercase=False,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0),
            ),
        ]
    )


def cross_val_scores(
    train: pd.DataFrame,
    pipeline: Pipeline | None = None,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Random cross-validation — kept to demonstrate the *wrong task*.

    Every prompt is answered by all 18 models, so a random split puts ~13.6
    of each validation passage's siblings into the training fold, each with a
    *different* label.  The classifier is asked to map one piece of content
    to 18 different authors, and the cheapest way to fit that is to ignore
    the content — and with it, the style.  The score collapses to 0.077 for
    words and 0.116 for characters, against 0.272 and 0.309 under
    prompt-grouped folds.  Use :func:`grouped_cv_scores` for anything you
    intend to believe.
    """
    pipeline = pipeline or build_word_pipeline()
    X = _as_text_array(train[TEXT_COL])
    y = train[TARGET_COL].to_numpy(dtype=object)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)
    return {
        "balanced_accuracy": balanced_accuracy_score(y, oof),
        "confusion": confusion_matrix(y, oof, labels=sorted(np.unique(y))),
        "labels": sorted(np.unique(y)),
        "oof": oof,
    }


def grouped_cv_scores(
    train: pd.DataFrame,
    pipeline: Pipeline | None = None,
    n_splits: int = 5,
    prompt_col: str = PROMPT_COL,
) -> dict:
    """Prompt-grouped cross-validation — the honest estimate.

    All the answers to one prompt land in the same fold, so the model can
    never learn a prompt's content from its siblings.
    """
    pipeline = pipeline or build_word_pipeline()
    X = _as_text_array(train[TEXT_COL])
    y = train[TARGET_COL].to_numpy(dtype=object)
    groups = train[prompt_col].to_numpy(dtype=object)
    cv = GroupKFold(n_splits=n_splits)
    oof = np.empty(len(y), dtype=object)
    for train_idx, valid_idx in cv.split(X, y, groups):
        pipeline.fit(X[train_idx], y[train_idx])
        oof[valid_idx] = pipeline.predict(X[valid_idx])
    return {
        "balanced_accuracy": balanced_accuracy_score(y, oof),
        "confusion": confusion_matrix(y, oof, labels=sorted(np.unique(y))),
        "labels": sorted(np.unique(y)),
        "oof": oof,
    }


def fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, pipeline: Pipeline | None = None
) -> np.ndarray:
    """Fit on train, predict labels for test."""
    pipeline = pipeline or build_word_pipeline()
    pipeline.fit(_as_text_array(train[TEXT_COL]), train[TARGET_COL].to_numpy(dtype=object))
    return pipeline.predict(_as_text_array(test[TEXT_COL]))

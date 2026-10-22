"""The stack: views -> prompt centring -> classifier -> group assignment.

Each layer exists because of a specific, measured failure of the layer
below it:

``views``
    Raw word n-grams spend most of their capacity on *topic*.  Adding
    topic-free views (function words, word shape) lifts group-CV balanced
    accuracy from 0.31 to 0.36.

``centring``
    Even the best view is still mostly measuring the prompt.  Subtracting
    the group mean removes the shared component, because what all 18 answers
    to one prompt have in common is the prompt, not the author.

``classifier``
    A linear model over the (centred) TF-IDF space.  Deliberately simple:
    it is the part of the system we understand best.

``assignment``
    Independent per-passage argmax throws away the knowledge that a prompt
    is answered by each model once.  Solving a small assignment problem
    inside each recovered group enforces it.

The layers are additive in the measurements that motivated them, and each
one can be switched off to reproduce the ablation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

from .decision import assign_within_groups, sinkhorn
from .groups import infer_groups
from .views import DEFAULT_VIEWS, fit_views, transform_views


@dataclass
class StackConfig:
    """Every knob of the stack, so that an ablation is one argument away."""

    views: tuple[str, ...] = DEFAULT_VIEWS
    C: float = 4.0
    center: bool = True
    assign: bool = True
    balance: bool = False
    group_k: int = 3
    group_cut: float = 0.88
    center_min_size: int = 2
    center_max_size: int | None = 18
    max_iter: int = 3000
    classifier: str = "logreg"
    random_state: int = 42
    #: "pool" groups the *whole* set of passages at once (what happens at
    #: test time); "fold" groups each cross-validation fold separately.
    grouping_scope: str = "pool"


def center_within(
    X: sparse.spmatrix,
    groups: np.ndarray,
    min_size: int = 2,
    max_size: int | None = 18,
) -> sparse.csr_matrix:
    """Subtract the mean of each group from every member of that group.

    This is the "the prompt is not the author" move made concrete: the mean
    of all the answers to one prompt is, to a first approximation, the
    prompt itself.  Removing it leaves the *residual* — what each model
    added on top of the shared material.

    Groups outside ``[min_size, max_size]`` are left untouched: a group of
    one carries no mean worth removing, and a group that merged two prompts
    would have its (meaningful) mean removed as well.
    """
    X = sparse.csr_matrix(X)
    groups = np.asarray(groups)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    keep = np.flatnonzero(counts >= min_size)
    if max_size is not None:
        keep = keep[counts[keep] <= max_size]
    if not len(keep):
        return X.tocsr()

    column = np.full(len(counts), -1)
    column[keep] = np.arange(len(keep))
    rows = np.flatnonzero(column[inverse] >= 0)
    membership = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, column[inverse][rows])),
        shape=(X.shape[0], len(keep)),
    )
    means = sparse.diags(1.0 / counts[keep]) @ (membership.T @ X)
    return (X - membership @ means).tocsr()


class StackModel:
    """The full pipeline, fit on labelled passages and applied to a pool."""

    def __init__(self, config: StackConfig | None = None):
        self.config = config or StackConfig()
        self.classes_: np.ndarray | None = None
        self.vecs_: dict | None = None
        self.clf_: object | None = None

    # -- fitting -----------------------------------------------------------
    def _make_classifier(self):
        cfg = self.config
        if cfg.classifier == "linearsvc":
            return LinearSVC(
                C=cfg.C,
                class_weight="balanced",
                max_iter=5000,
                random_state=cfg.random_state,
            )
        return LogisticRegression(
            C=cfg.C,
            max_iter=cfg.max_iter,
            class_weight="balanced",
            random_state=cfg.random_state,
        )

    def fit(
        self,
        texts: Sequence[str],
        y: Sequence[str],
        groups: Sequence | None = None,
    ) -> StackModel:
        """Fit the stack.

        *groups* gives the known prompt grouping of the training passages
        (the train set's ``prompt_id`` column).  When omitted, the grouping
        is inferred from the texts exactly as it is at prediction time.
        """
        cfg = self.config
        self.vecs_ = fit_views(texts, cfg.views)
        X = transform_views(self.vecs_, texts, cfg.views)
        if cfg.center:
            if groups is None:
                groups = infer_groups(texts, k=cfg.group_k, cut=cfg.group_cut)
            X = center_within(X, np.asarray(groups), cfg.center_min_size, cfg.center_max_size)
        self.clf_ = self._make_classifier()
        self.clf_.fit(X, np.asarray(y, dtype=object))
        self.classes_ = np.asarray(self.clf_.classes_, dtype=object)
        return self

    # -- prediction --------------------------------------------------------
    def decision_function(
        self, texts: Sequence[str], groups: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scores for every class, plus the group id of every passage."""
        cfg = self.config
        assert self.vecs_ is not None and self.clf_ is not None, "fit first"
        X = transform_views(self.vecs_, texts, cfg.views)
        if groups is None:
            groups = infer_groups(texts, k=cfg.group_k, cut=cfg.group_cut)
        groups = np.asarray(groups)
        if cfg.center:
            X = center_within(X, groups, cfg.center_min_size, cfg.center_max_size)
        # The classifier's own class order is the column order of ``classes_``
        # (both come from ``clf_.classes_``), so the scores need no remapping.
        scores = np.asarray(self.clf_.decision_function(X), dtype=float)
        if scores.ndim == 1:  # binary fallback
            scores = np.column_stack([-scores, scores])
        return scores, groups

    def predict(self, texts: Sequence[str], groups: np.ndarray | None = None) -> np.ndarray:
        """Label every passage in the pool."""
        cfg = self.config
        scores, groups = self.decision_function(texts, groups)
        if cfg.balance:
            probs = _softmax(scores)
            scores = np.log(np.maximum(sinkhorn(probs), 1e-12))
        if cfg.assign:
            return assign_within_groups(scores, groups, self.classes_)
        return self.classes_[scores.argmax(axis=1)]


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -60, 0))
    return exp / exp.sum(axis=1, keepdims=True)


def group_cv_score(
    texts: Sequence[str],
    y: Sequence[str],
    groups: Sequence,
    config: StackConfig | None = None,
    n_splits: int = 5,
    return_oof: bool = False,
) -> dict:
    """Out-of-fold balanced accuracy under ``GroupKFold`` on the prompt ids.

    Random cross-validation is *not* a proxy for the leaderboard here — it is
    a different and much harder task.  Siblings of a validation passage (the
    other answers to the same prompt) sit in the training fold carrying
    *different* labels, so the model is asked to map one piece of content to
    18 different authors; the cheapest way to fit that is to ignore the
    content, and with it the style.  Grouping by prompt removes the
    contradiction and lets the model use style instead.

    The *validation* grouping is recovered from text alone, never from
    ``groups``.  With ``config.grouping_scope == "pool"`` the recovery runs
    over the whole set of passages, exactly as it will over the whole test
    pool — which matters, because grouping quality depends on pool size
    (see :mod:`ntl.groups`).
    """
    cfg = config or StackConfig()
    y = np.asarray(y, dtype=object)
    groups = np.asarray(groups)
    oof = np.empty(len(y), dtype=object)
    folds: list[float] = []

    pool_groups = None
    if cfg.grouping_scope == "pool" and (cfg.center or cfg.assign):
        pool_groups = infer_groups(texts, k=cfg.group_k, cut=cfg.group_cut)

    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, valid_idx in splitter.split(texts, y, groups):
        model = StackModel(cfg)
        model.fit(
            [texts[i] for i in train_idx],
            y[train_idx],
            groups=groups[train_idx],
        )
        if pool_groups is not None:
            valid_groups = pool_groups[valid_idx]
        elif cfg.center or cfg.assign:
            valid_groups = infer_groups(
                [texts[i] for i in valid_idx], k=cfg.group_k, cut=cfg.group_cut
            )
        else:
            valid_groups = np.zeros(len(valid_idx), dtype=int)
        pred = model.predict([texts[i] for i in valid_idx], groups=valid_groups)
        oof[valid_idx] = pred
        folds.append(balanced_accuracy_score(y[valid_idx], pred))

    result = {
        "balanced_accuracy": float(balanced_accuracy_score(y, oof)),
        "fold_scores": folds,
        "fold_mean": float(np.mean(folds)),
        "fold_std": float(np.std(folds)),
    }
    if return_oof:
        result["oof"] = oof
    return result

"""Which facets actually carry author signal?

A facet can look informative for the wrong reason.  Response length, for
example, has a large mutual information with the model label on the train set
— because some prompts ask for an essay and others for a sentence, and the
prompt mix is not uniform across models by accident.  That is a *prompt
proxy*, not an author habit, and it will not transfer to a hidden prompt.

The honest question is: **after the prompt is held constant, does the facet
still vary with the model?**  This module answers it with two statistics and
one classifier.

``eta2_within``
    Centre each prompt group on its own mean, then ask how much of the
    remaining variance the model label explains (a one-way ANOVA on the
    residuals).  Near zero means the facet is a prompt proxy; clearly above
    zero means it separates siblings writing about the same thing.

``mi``
    Raw mutual information with the label.  Reported only as a contrast: a
    facet with high ``mi`` and low ``eta2_within`` is the trap described
    above.  It is never used for selection.

``facet_cv_score``
    Prompt-grouped cross-validation with the facet block alone.  This is the
    end-to-end check: does the signal survive a fitted classifier?

``infer_domains``
    Clusters recovered-group centroids into topics.  Included to *demonstrate*
    the prompt-side constraint rather than to build a feature: the inferred
    domain is constant inside every recovered group, so it cannot separate
    siblings (see :func:`group_constant_share`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, normalize

from . import facets
from .pipeline import center_dense_within

#: ``eta2_within`` below this is treated as "no author signal".  It is a
#: reporting threshold, not a fitted parameter: the notebook prints the whole
#: ranking so the reader can disagree with the cut.
DEFAULT_SIGNAL_FLOOR = 0.005


def within_group_eta2(
    values: Sequence[float],
    y: Sequence[str],
    groups: Sequence,
    min_group: int = 2,
) -> float:
    """Share of within-group variance explained by the model label.

    1. subtract each group's mean (removing the prompt);
    2. run a one-way ANOVA of the residuals by model;
    3. return between-model SS / total SS.

    Groups smaller than *min_group* are left in place: they contribute no
    within-group variation, so removing them would silently flatter the
    result.
    """
    v = np.asarray(values, dtype=float)
    y = np.asarray(y, dtype=object)
    groups = np.asarray(groups)
    if len(v) < 3:
        return 0.0

    frame = pd.DataFrame({"v": v, "g": groups})
    centred = v - frame.groupby("g")["v"].transform("mean").to_numpy(dtype=float)
    total = float(np.sum((centred - centred.mean()) ** 2))
    if total < 1e-12:
        return 0.0

    between = 0.0
    for label in np.unique(y):
        mask = y == label
        if mask.sum() == 0:
            continue
        between += float(mask.sum()) * float(centred[mask].mean() ** 2)
    return float(np.clip(between / total, 0.0, 1.0))


def mutual_information(values: Sequence[float], y: Sequence[str]) -> float:
    """Raw mutual information between a facet and the label.

    Discretisation is automatic.  High values are *not* evidence of an author
    habit — see the module docstring.
    """
    v = np.asarray(values, dtype=float).reshape(-1, 1)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return float(mutual_info_classif(v, np.asarray(y, dtype=object), random_state=0)[0])


def group_constant_share(values: Sequence[float], groups: Sequence) -> float:
    """Share of groups in which the facet takes a single value.

    A facet at 1.0 is constant inside every prompt group and therefore
    *cannot* identify the author no matter how large its raw mutual
    information is.  Domain labels and prompt-level axes land here.
    """
    v = np.asarray(values)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if not len(unique):
        return 0.0
    constant = 0
    for group in unique:
        if len(np.unique(v[groups == group])) <= 1:
            constant += 1
    return constant / len(unique)


def _mean_within_sd(values: np.ndarray, groups: np.ndarray) -> float:
    sds = []
    for group in np.unique(groups):
        block = values[groups == group]
        if len(block) >= 2:
            sds.append(float(np.std(block)))
    return float(np.mean(sds)) if sds else 0.0


def facet_signal(
    texts: Sequence[str],
    y: Sequence[str],
    groups: Sequence,
    signal_floor: float = DEFAULT_SIGNAL_FLOOR,
) -> pd.DataFrame:
    """One row per facet: ``mi``, ``eta2_within``, and a verdict.

    The frame is sorted by ``eta2_within`` descending, so the first rows are
    the facets worth putting in a classifier.
    """
    y = np.asarray(y, dtype=object)
    groups = np.asarray(groups)
    base = facets.response_facets(texts)
    frame = pd.concat([base, facets.group_facets(texts, groups, base)], axis=1)

    rows = []
    for column in frame.columns:
        values = np.nan_to_num(frame[column].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        eta2 = within_group_eta2(values, y, groups)
        rows.append(
            {
                "facet": column,
                "family": facets.family_of(column),
                "mi": mutual_information(values, y),
                "eta2_within": eta2,
                "constant_share": group_constant_share(values, groups),
                "within_sd": _mean_within_sd(values, groups),
                "verdict": "author signal" if eta2 >= signal_floor else "prompt proxy",
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("eta2_within", ascending=False).reset_index(drop=True)


def _facet_matrix(texts: Sequence[str], groups: np.ndarray, families: Sequence[str]) -> np.ndarray:
    frame = facets.facet_frame(texts, groups, families).to_numpy(dtype=float)
    return np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)


def facet_cv_score(
    texts: Sequence[str],
    y: Sequence[str],
    groups: Sequence,
    families: Sequence[str],
    n_splits: int = 5,
    C: float = 4.0,
    center: bool = True,
) -> dict:
    """Prompt-grouped CV using the facet block alone.

    The numeric block is standardised and (by default) centred within the
    true prompt groups.  Because ``GroupKFold`` never splits a prompt, using
    the true grouping here is the transductive condition, not a leak — at test
    time the grouping is *recovered*, which is slightly noisier (ARI 0.97),
    so treat the result as a mild upper bound on the facet's contribution.
    """
    y = np.asarray(y, dtype=object)
    groups = np.asarray(groups)
    X = _facet_matrix(texts, groups, families)

    splitter = GroupKFold(n_splits=n_splits)
    oof = np.empty(len(y), dtype=object)
    folds: list[float] = []
    for train_idx, valid_idx in splitter.split(X, y, groups):
        scaler = StandardScaler().fit(X[train_idx])
        X_train = scaler.transform(X[train_idx])
        X_valid = scaler.transform(X[valid_idx])
        if center:
            X_train = center_dense_within(X_train, groups[train_idx])
            X_valid = center_dense_within(X_valid, groups[valid_idx])
        clf = LogisticRegression(
            C=C,
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X_train, y[train_idx])
        pred = clf.predict(X_valid)
        oof[valid_idx] = pred
        folds.append(float(balanced_accuracy_score(y[valid_idx], pred)))

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, oof)),
        "fold_scores": folds,
        "fold_mean": float(np.mean(folds)),
        "fold_std": float(np.std(folds)),
    }


def family_ranking(
    texts: Sequence[str],
    y: Sequence[str],
    groups: Sequence,
    families: Sequence[str] | None = None,
    n_splits: int = 5,
    C: float = 4.0,
) -> pd.DataFrame:
    """Grouped CV for each facet family on its own, best first."""
    if families is None:
        families = [*facets.RESPONSE_FAMILIES, "group"]
    rows = []
    for family in families:
        score = facet_cv_score(texts, y, groups, [family], n_splits=n_splits, C=C)
        rows.append(
            {
                "family": family,
                "n_facets": len(facets.facet_columns([family])),
                "balanced_accuracy": score["balanced_accuracy"],
                "fold_mean": score["fold_mean"],
                "fold_std": score["fold_std"],
            }
        )
    return (
        pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False).reset_index(drop=True)
    )


def infer_domains(
    texts: Sequence[str],
    groups: Sequence,
    n_topics: int = 6,
    random_state: int = 42,
) -> dict:
    """Cluster recovered-group centroids into topics.

    Returns ``{"passage_domain": Series, "group_domain": Series,
    "n_topics": int}``.  The point is not the topic model — it is that
    ``group_constant_share(passage_domain, groups) == 1.0``: whatever the
    domain is, every passage in a group gets the same value, so the domain
    cannot separate siblings.  It can only stratify *training*.
    """
    groups = np.asarray(groups)
    n = len(texts)
    unique = np.unique(groups)
    membership = pd.Series(np.arange(len(unique)), index=unique)
    inverse = membership.loc[groups].to_numpy()

    vectorizer = TfidfVectorizer(
        stop_words="english", sublinear_tf=True, min_df=3, dtype=np.float32
    )
    X = vectorizer.fit_transform(texts)
    indicator = sparse.csr_matrix((np.ones(n), (np.arange(n), inverse)), shape=(n, len(unique)))
    centroids = normalize(indicator.T @ X)

    k = int(min(n_topics, len(unique)))
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    group_domain = model.fit_predict(centroids.toarray())
    passage_domain = group_domain[inverse]
    return {
        "passage_domain": pd.Series(passage_domain, name="domain"),
        "group_domain": pd.Series(group_domain, index=unique, name="domain"),
        "n_topics": k,
    }

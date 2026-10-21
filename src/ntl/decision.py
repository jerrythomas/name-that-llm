"""Decision rules — turning scores into labels, and what the metric wants.

The competition metric is **balanced accuracy**, the mean of per-class
recalls::

    balanced_accuracy = (1/K) * sum_k  #{i : y_i = k and yhat_i = k} / N_k

where ``N_k`` is the number of test passages whose true label is ``k``.  Two
consequences drive everything in this module.

**1. What is the optimal rule for one passage?**
Write the metric as a sum over passages::

    (1/K) * sum_i sum_k P(y_i = k | x_i) * 1[yhat_i = k] / N_k

Every passage can be labelled independently, and the expected contribution of
labelling passage *i* with class *j* is ``P(y_i = j | x_i) / N_j``.  So the
metric-optimal choice is::

    yhat_i = argmax_j P(y_i = j | x_i) / N_j

If the classifier was trained on a *uniform* class prior (this train set is
uniform: 150 passages per model) and the test prior is ``pi_test``, then
``P(y = j | x) ∝ p_source(j | x) * pi_test_j / pi_source_j`` and ``N_j`` is
proportional to ``pi_test_j``.  The prior terms cancel::

    yhat_i = argmax_j p_source(j | x_i) / pi_source_j

With a uniform source prior that is simply **plain argmax** of the model
scores, whatever the test class mix happens to be.  Two corollaries, both
verified empirically on the train set:

* *Do not* try to reproduce the true test class mix (Sinkhorn towards the
  observed mix): it costs a lot of accuracy under any skew.
* *Do not* divide by an estimated test prior either: that is the correction
  applied in the wrong direction, and it is worse still.

The train prior being uniform is doing the work here.  That is worth stating
plainly, because it is the reason the pipeline deliberately does **not**
rebalance its output.

**2. What about the structure inside a prompt?**
Independent argmax ignores a piece of hard information: in the train set
each prompt is answered by each of the 18 models exactly once.  Inside a
recovered group the labels should therefore be a *permutation*, not 18
independent draws.  Enforcing that turns 18 independent decisions into one
joint one and is worth roughly +0.10 balanced accuracy.  The constraint is
about the *design of the data*, not about the class mix — and it is applied
as "at most one passage per model per group", which stays valid even if the
hidden test is not perfectly balanced (measured: it still helps, by less).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def argmax_decision(scores: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    """Plain argmax — the metric-optimal rule for a uniform source prior."""
    return np.asarray(classes, dtype=object)[np.asarray(scores).argmax(axis=1)]


def sinkhorn(
    probabilities: np.ndarray, target: np.ndarray | None = None, iters: int = 20
) -> np.ndarray:
    """Iterative proportional fitting of the column marginals to *target*.

    With ``target=None`` this pushes the predicted class mass towards a
    uniform mix.  It is a *heuristic* about the output distribution, not a
    consequence of balanced accuracy — see the module docstring.  It was
    measured to give a small, mix-robust improvement over plain argmax, but
    it is not what the metric asks for, so it is opt-in.
    """
    P = np.array(probabilities, dtype=float, copy=True)
    n, k = P.shape
    if target is None:
        target = np.full(k, n / k)
    target = np.asarray(target, dtype=float)
    for _ in range(iters):
        P *= target / np.maximum(P.sum(axis=0), 1e-12)
        P /= np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    return P


def em_prior(
    probabilities: np.ndarray,
    source_prior: np.ndarray | None = None,
    iters: int = 50,
    tol: float = 1e-9,
) -> np.ndarray:
    """Saerens-Latinne-Decaestecker estimate of the *test* class prior.

    E-step: re-weight the source posterior by the prior shift.
    M-step: the prior is the mean re-weighted posterior.

    Included because it is the standard tool for this question — and
    included with a warning: for **balanced accuracy** the test prior is
    *not* what the decision rule should divide by (the module docstring
    derives why).  Applying it costs most of the accuracy.
    """
    P = np.asarray(probabilities, dtype=float)
    k = P.shape[1]
    src = np.full(k, 1.0 / k) if source_prior is None else np.asarray(source_prior, float)
    pi = src.copy()
    for _ in range(iters):
        W = P * (pi / src)
        W /= np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
        new = W.mean(axis=0)
        if np.abs(new - pi).max() < tol:
            pi = new
            break
        pi = new
    return pi


def assign_within_groups(
    scores: np.ndarray,
    groups: np.ndarray,
    classes: Sequence[str],
    at_most_one: bool = True,
) -> np.ndarray:
    """Assign labels inside each group so that no model repeats (if possible).

    Parameters
    ----------
    scores:
        ``(n_passages, n_classes)`` score matrix (higher is better).
    groups:
        Group id per passage.
    classes:
        Class names, in the order of ``scores``' columns.
    at_most_one:
        When True (default) a model may be used at most once per group.  This
        is the relaxation that stays valid even if the hidden test does not
        contain exactly one passage per model per prompt.  When False every
        passage is labelled independently by argmax.

    Returns
    -------
    Array of labels, one per passage.
    """
    scores = np.asarray(scores, dtype=float)
    classes = np.asarray(classes, dtype=object)
    groups = np.asarray(groups)
    n_classes = scores.shape[1]
    labels = np.empty(len(scores), dtype=object)

    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        block = scores[idx]
        if not at_most_one or len(idx) == 1:
            labels[idx] = classes[block.argmax(axis=1)]
            continue
        if len(idx) <= n_classes:
            rows, cols = linear_sum_assignment(-block)
            assigned = np.empty(len(idx), dtype=object)
            assigned[rows] = classes[cols]
            labels[idx] = assigned
        else:
            # More passages than models: by pigeonhole some model must repeat.
            # Pad with dummy columns so every passage still receives a label
            # and real models stay unique; passages forced onto a dummy column
            # keep their own best guess.
            padding = np.full((len(idx), len(idx) - n_classes), -1e9)
            padded = np.concatenate([block, padding], axis=1)
            rows, cols = linear_sum_assignment(-padded)
            fallback = classes[block.argmax(axis=1)]
            chosen = np.where(cols < n_classes, classes[np.clip(cols, 0, n_classes - 1)], fallback)
            assigned = np.empty(len(idx), dtype=object)
            assigned[rows] = chosen[rows]
            labels[idx] = assigned
    return labels


def balanced_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Mean per-class recall — the competition metric."""
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(list(y_true), list(y_pred)))

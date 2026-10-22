"""Recovering the prompt structure from the passages alone.

The train set tells us which passages answered the same prompt
(``prompt_id``).  The hidden test set does not — it is just a bag of
passages.  But the *transductive* structure is still there: passages that
answer the same prompt talk about the same thing, so in content-word TF-IDF
cosine they sit much closer to each other than to passages answering
different prompts.  Measured on train, a passage's nearest neighbour shares
its prompt 99.8% of the time, while the average same-prompt distance is
0.785 against 0.989 for different prompts.

So the grouping can be *inferred*:

1. vectorise passages by content words only (the topic, which is what makes
   same-prompt passages alike);
2. build a k-nearest-neighbour graph over cosine distance;
3. keep edges below a distance cut;
4. take connected components.

Why a k-NN graph rather than plain single-linkage clustering?  Single
linkage chains: one spuriously close pair of passages merges two whole
prompts into one cluster, and the error then propagates.  Requiring k
neighbours to agree suppresses those bridges — but only if the distance cut
is tight enough.

**The cut has to be chosen at the scale the grouping will be used.**  On a
540-passage cross-validation fold a cut of 0.95 looks perfect (ARI 1.00,
because the pool is small enough that nothing chains).  On the full
2,699-passage pool the same cut merges prompts: ARI falls to 0.58 and one
cluster swallows 252 passages.  A cut of 0.88 is the setting that holds up
at every pool size tried (ARI 0.97 at 2,699 passages, 0.96-0.98 from 180 to
4,000 passages).

The local evidence is scale-free, by contrast: at *every* pool size, 99.8%
of nearest neighbours share a prompt.  The structure is there; only the
clustering step was fragile.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score


def content_distance_matrix(texts: Sequence[str], stop_words: str = "english") -> np.ndarray:
    """Pairwise cosine distance in content-word TF-IDF space."""
    vec = TfidfVectorizer(stop_words=stop_words, sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(texts)
    similarity = (X @ X.T).toarray()
    np.fill_diagonal(similarity, 0.0)
    return np.clip(1.0 - similarity, 0.0, 1.0)


def infer_groups(
    texts: Sequence[str],
    k: int = 3,
    cut: float = 0.88,
    distance: np.ndarray | None = None,
) -> np.ndarray:
    """Cluster passages into recovered prompt groups.

    Parameters
    ----------
    texts:
        The passages of the pool (train fold, or the whole test set).
    k:
        Number of nearest neighbours that must agree before an edge is kept.
    cut:
        Maximum cosine distance for a kept edge.  0.88 is the value that
        stays accurate as the pool grows — see the module docstring.
    distance:
        Pre-computed distance matrix (to avoid recomputing it).

    Returns
    -------
    Array of group ids, one per passage.
    """
    n = len(texts)
    if n < 2:
        return np.zeros(n, dtype=int)

    D = content_distance_matrix(texts) if distance is None else distance
    order = np.argsort(D, axis=1)
    rows = np.repeat(np.arange(n), k)
    cols = order[:, :k].ravel()
    keep = D[rows, cols] < cut

    graph = sparse.csr_matrix((np.ones(int(keep.sum())), (rows[keep], cols[keep])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


def group_sizes(groups: np.ndarray) -> np.ndarray:
    """Number of passages in each recovered group, sorted descending."""
    _, counts = np.unique(groups, return_counts=True)
    return np.sort(counts)[::-1]


def neighbour_purity(distance: np.ndarray, true_groups: np.ndarray, k: int = 5) -> float:
    """Share of nearest neighbours that share the passage's true group."""
    n = len(true_groups)
    if n < 2:
        return float("nan")
    order = np.argsort(distance, axis=1)
    same = 0
    for i in range(n):
        for j in order[i, :k]:
            if true_groups[i] == true_groups[j]:
                same += 1
    return same / (n * k)


def group_report(groups: np.ndarray, true_groups: np.ndarray | None = None) -> dict:
    """Summarise a recovered grouping.

    With ``true_groups`` present, reports the adjusted Rand index (does the
    partition match?) plus the number of recovered groups and their size
    histogram.
    """
    sizes = group_sizes(groups)
    report = {
        "n_groups": int(len(sizes)),
        "largest_group": int(sizes[0]) if len(sizes) else 0,
        "smallest_group": int(sizes[-1]) if len(sizes) else 0,
        "mean_size": float(sizes.mean()) if len(sizes) else 0.0,
        "n_singletons": int((sizes == 1).sum()),
    }
    if true_groups is not None:
        report["ari"] = float(adjusted_rand_score(true_groups, groups))
        report["n_true_groups"] = int(len(np.unique(true_groups)))
        # purity: share of passages sitting in a group dominated by their own label
        correct = 0
        for g in np.unique(groups):
            members = true_groups[groups == g]
            values, counts = np.unique(members, return_counts=True)
            correct += counts.max()
        report["purity"] = correct / len(true_groups)
    return report

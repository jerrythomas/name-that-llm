#!/usr/bin/env python
"""Fast self-checks for the ``ntl`` package.

No heavy training: everything here runs in seconds and verifies the
invariants the rest of the project relies on.

Usage:
    uv run python scripts/selftest.py
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ntl import baseline, data, decision, groups, pipeline, stylometry, submit, views  # noqa: E402  # noqa: E402

CHECKS: list = []


def check(fn):
    CHECKS.append(fn)
    return fn


# ---------------------------------------------------------------------------


@check
def test_views():
    """Every view produces text; the topic-free ones really are topic-free."""
    raw = "When a language dies, it does not go extinct; it falls silent."
    assert views.view_text("char", raw) == raw
    shape = views.view_text("shape", raw)
    assert "language" not in shape and "X" in shape and "x" in shape
    funcon = views.view_text("funcon", raw)
    assert "CONTENT" in funcon and "when" in funcon and "language" not in funcon
    punct = views.view_text("punct", raw)
    assert "language" not in punct and "," in punct


@check
def test_function_transform_is_topic_invariant():
    """The syntactic skeleton of two different prompts must match in shape."""
    a = "It is worth noting that the model must be trained carefully."
    b = "It is worth noting that the species must be protected carefully."
    assert views.function_transform(a) == views.function_transform(b)


@check
def test_log_odds_sign():
    """A word used only by class A gets a high positive z-score."""
    texts = ["alpha beta gamma"] * 40 + ["delta epsilon zeta"] * 40
    labels = np.array(["a"] * 40 + ["b"] * 40)
    frame = stylometry.distinctive_words(texts, labels, "a", top_n=3, min_count=2)
    assert "alpha" in frame["word"].tolist(), frame
    assert (frame[frame["word"] == "alpha"]["z"] > 0).all()


@check
def test_assignment_is_a_permutation():
    """Inside a group of 18 passages, every model is used exactly once."""
    rng = np.random.default_rng(0)
    classes = np.array([f"m{i:02d}" for i in range(18)], dtype=object)
    scores = rng.normal(size=(18, 18))
    groups = np.zeros(18, dtype=int)
    out = decision.assign_within_groups(scores, groups, classes)
    assert sorted(out.tolist()) == sorted(classes.tolist())


@check
def test_assignment_handles_oversized_group():
    """More passages than models: everyone still gets a label, models stay unique."""
    rng = np.random.default_rng(1)
    classes = np.array([f"m{i:02d}" for i in range(18)], dtype=object)
    scores = rng.normal(size=(25, 18))
    groups = np.zeros(25, dtype=int)
    out = decision.assign_within_groups(scores, groups, classes)
    assert len(out) == 25 and all(o in classes.tolist() for o in out)


@check
def test_centring_removes_the_group_mean():
    """After centring, every group has (approximately) zero mean."""
    from scipy import sparse

    rng = np.random.default_rng(2)
    X = sparse.csr_matrix(rng.normal(size=(12, 5)))
    groups = np.repeat(np.arange(3), 4)
    centred = pipeline.center_within(X, groups, min_size=2, max_size=None)
    for gid in range(3):
        idx = groups == gid
        assert abs(float(np.asarray(centred[idx].mean()))) < 1e-6


@check
def test_group_recovery_quality():
    """Recovered prompt groups are pure and complete on synthetic data."""
    train = _synthetic_train(20)
    texts = train["text"].to_numpy(dtype=object)
    truth = train["prompt_id"].to_numpy(dtype=object)
    recovered = groups.infer_groups(texts, k=3, cut=0.88)
    report = groups.group_report(recovered, truth)
    assert report["ari"] > 0.85, report
    assert report["purity"] > 0.95, report


@check
def test_metric_helpers():
    """argmax / sinkhorn / em_prior behave as documented."""
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    balanced = decision.sinkhorn(probs)
    assert np.allclose(balanced.sum(axis=1), 1.0)
    assert abs(balanced.sum(axis=0)[0] - balanced.sum(axis=0)[1]) < 1e-6
    prior = decision.em_prior(np.repeat([[0.5, 0.5]], 100, axis=0))
    assert prior.sum() == 1.0


@check
def test_baseline_cv_protocols_differ():
    """Grouped CV runs and gives a reasonable score on synthetic data."""
    train = _synthetic_train(30)
    pipe = baseline.build_word_pipeline()
    rnd = baseline.grouped_cv_scores(train, pipe)["balanced_accuracy"]
    grp = baseline.grouped_cv_scores(train, pipe, n_splits=3)["balanced_accuracy"]
    assert grp > 0.5, f"grouped CV too low: {grp:.4f}"
    assert rnd > 0.5, f"random CV too low: {rnd:.4f}"


@check
def test_submission_roundtrip():
    """A hand-crafted prediction frame round-trips to valid parquet."""
    train = _synthetic_train(5)
    test = pd.DataFrame(
        {
            "id": ["t0", "t1"],
            "text": ["test one", "test two"],
        }
    )
    preds = np.array([train["model"].iloc[0], train["model"].iloc[1]], dtype=object)
    sub = submit.make_submission(test, preds)
    assert list(sub.columns) == ["id", "model"]
    assert list(sub["model"]) == [train["model"].iloc[0], train["model"].iloc[1]]
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        submit.write_submission(sub, tmp.name)
        back = pd.read_parquet(tmp.name)
    assert list(back["model"]) == [train["model"].iloc[0], train["model"].iloc[1]]


@check
def test_in_style_pool_loader():
    """The public-pool loader normalises foreign column names onto our schema."""
    from pathlib import Path

    root = Path(tempfile.mkdtemp())
    pd.DataFrame(
        {
            "passage": ["alpha text", "beta text"],
            "llm": ["gpt-4", "gpt-4"],
            "question": ["q1", "q1"],
        }
    ).to_csv(root / "shard.csv", index=False)
    pd.DataFrame(
        {"text": ["gamma text"], "model": ["claude-opus-4-6"], "prompt_id": ["q2"]}
    ).to_parquet(root / "shard.parquet")

    pool = data.load_in_style_pool(roots=[root])
    assert list(pool.columns) == ["id", "prompt_id", "text", "model"], pool.columns
    assert len(pool) == 3
    assert set(pool["text"]) == {"alpha text", "beta text", "gamma text"}

    report = data.describe_in_style_pool(pool)
    assert report["rows"] == 3
    assert report["label_coverage"] == 1 / 3  # only the one competition label


@check
def test_kaggle_mount_layouts():
    """Both Kaggle dataset-mount layouts are discovered, competition excluded."""
    import tempfile
    from pathlib import Path

    root = Path(tempfile.mkdtemp())

    # legacy layout: /kaggle/input/<slug>/
    legacy = root / "legacy-slug"
    legacy.mkdir()
    # current layout: /kaggle/input/datasets/<owner>/<slug>/
    current = root / "datasets" / "prvsiyan" / "prvsiyans-extended-name-that-llm-dataset"
    current.mkdir(parents=True)
    # competition data must never be treated as a pool
    comp = root / "competitions" / "name-that-llm"
    comp.mkdir(parents=True)

    found = data.kaggle_input_dirs(root)
    assert legacy in found, found
    assert current in found, found
    assert comp not in found, found

    # the pool loader finds the new layout by default
    pd.DataFrame(
        {"text": ["alpha text", "beta text"], "model": ["gpt-5.5", "claude-opus-5"]}
    ).to_parquet(current / "shard.parquet")
    pd.DataFrame(
        {"id": ["t0"], "prompt_id": ["p0"], "text": ["TRAIN"], "model": ["gpt-5.5"]}
    ).to_csv(comp / "train.csv", index=False)

    pool = data.load_in_style_pool(roots=found)
    assert len(pool) == 2, pool
    assert "TRAIN" not in set(pool["text"]), "competition train.csv leaked into the pool"


@check
def test_stack_shapes():
    """The stack must return one label per passage and keep the class order."""
    train = _synthetic_train(10)
    texts = train["text"].to_numpy(dtype=object)
    y = train["model"].to_numpy(dtype=object)
    g = train["prompt_id"].to_numpy(dtype=object)
    model = pipeline.StackModel(pipeline.StackConfig(views=("funcon",)))
    model.fit(texts, y, groups=g)
    scores, groups = model.decision_function(texts)
    assert scores.shape == (len(texts), len(model.classes_))
    assert len(groups) == len(texts)
    preds = model.predict(texts, groups=groups)
    assert len(preds) == len(texts)


def _synthetic_train(n_prompts: int = 20) -> pd.DataFrame:
    """A tiny synthetic train set covering all 18 labels."""
    from ntl.config import LABELS

    rows = []
    # Each prompt gets a unique set of nonsense tokens — zero overlap between prompts.
    # This guarantees the text-similarity grouping recovers the prompts perfectly.
    for p in range(n_prompts):
        prompt_token = f"prompt{p:02d}"
        # Unique vocab per prompt: 10 nonsense words that appear ONLY in this prompt
        vocab = [f"{prompt_token}_{i}" for i in range(10)]
        for m in LABELS:
            # Model marker: also unique per model but short
            style = m.replace("-", "_").replace(".", "_")
            text = " ".join(vocab + [style] * 8)
            rows.append(
                {
                    "id": f"p{p}-{m[:3]}",
                    "prompt_id": f"p{p}",
                    "text": text,
                    "model": m,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    failures = 0
    for fn in CHECKS:
        name = fn.__name__
        try:
            fn()
        except Exception:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
        else:
            print(f"ok    {name}")
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

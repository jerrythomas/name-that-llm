"""Plotting helpers for exploration.

Each function returns a matplotlib ``(fig, ax)`` so notebooks can further
customize, and optionally saves to ``outputs/``. matplotlib/seaborn are
imported inside functions to keep imports light.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import LABELS, PROVIDERS


def _finish(fig, ax, save_path: str | Path | None):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig, ax


def _provider_colors(models: list[str]) -> list[str]:
    palette = {
        "OpenAI": "#10a37f",
        "Google": "#4285f4",
        "Anthropic": "#d97757",
        "Qwen": "#7c3aed",
        "ZAI": "#0891b2",
        "Moonshot AI": "#e11d48",
    }
    return [palette.get(PROVIDERS.get(m, ""), "#888888") for m in models]


def plot_label_distribution(
    dist: pd.DataFrame, save_path: str | Path | None = None
):
    """Bar chart of passages per model, coloured by provider.

    *dist* is the output of :func:`ntl.explore.label_distribution`.
    """
    import matplotlib.pyplot as plt

    d = dist.sort_values("n_passages", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(d["model"], d["n_passages"], color=_provider_colors(list(d["model"])))
    for y, (n, share) in enumerate(zip(d["n_passages"], d["share"], strict=False)):
        ax.text(n + 2, y, f"{n}  ({share:.1%})", va="center", fontsize=8)
    ax.set_xlabel("passages in train.csv")
    ax.set_title("Class balance across the 18 models")
    ax.margins(x=0.15)
    return _finish(fig, ax, save_path)


def plot_prompt_coverage(pstats: pd.DataFrame, save_path: str | Path | None = None):
    """Histogram of distinct models per prompt (how complete each prompt is)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(pstats["n_models"], bins=range(1, 20), color="#4c72b0", edgecolor="white")
    ax.set_xlabel("distinct models answering the prompt")
    ax.set_ylabel("number of prompts")
    ax.set_title("Prompt coverage: how many of the 18 models answered each prompt")
    return _finish(fig, ax, save_path)


def plot_length_by_model(
    train: pd.DataFrame, save_path: str | Path | None = None
):
    """Box plot of word counts per model (competition says 40-250 words)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(11, 7))
    order = LABELS[::-1]
    d = train.assign(n_words=train["text"].str.split().str.len())
    sns.boxplot(
        data=d,
        y="model",
        x="n_words",
        order=order,
        hue="model",
        hue_order=order,
        palette=_provider_colors(order),
        legend=False,
        ax=ax,
    )
    ax.set_title("Word-count distribution per model")
    ax.set_xlabel("words per passage")
    ax.set_ylabel("")
    return _finish(fig, ax, save_path)


def plot_feature_heatmap(
    zscores: pd.DataFrame, save_path: str | Path | None = None
):
    """Heatmap of z-scored style features per model (rows = models)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        zscores,
        cmap="RdBu_r",
        center=0,
        annot=False,
        cbar_kws={"label": "z-score vs other models"},
        ax=ax,
    )
    ax.set_title("Stylistic fingerprint per model (mean features, z-scored)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    return _finish(fig, ax, save_path)


def plot_length_scatter(train: pd.DataFrame, save_path: str | Path | None = None):
    """Words vs characters, coloured by provider — sanity check on length."""
    import matplotlib.pyplot as plt

    d = train.assign(
        n_words=train["text"].str.split().str.len(),
        n_chars=train["text"].str.len(),
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    for provider in sorted(set(PROVIDERS.values())):
        models = [m for m, p in PROVIDERS.items() if p == provider]
        sub = d[d["model"].isin(models)]
        ax.scatter(
            sub["n_words"], sub["n_chars"], s=8, alpha=0.4, label=provider
        )
    ax.set_xlabel("words")
    ax.set_ylabel("characters")
    ax.set_title("Passage length: words vs characters")
    ax.legend(title="provider", fontsize=8)
    return _finish(fig, ax, save_path)


def plot_distinctive_tokens(
    tokens: dict[str, pd.DataFrame],
    models: list[str] | None = None,
    save_path: str | Path | None = None,
):
    """Grid of top distinctive tokens per model (one small bar chart each)."""
    import matplotlib.pyplot as plt

    models = models or list(tokens.keys())
    ncols = 3
    nrows = -(-len(models) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows))
    axes = axes.ravel()
    for ax, model in zip(axes, models, strict=False):
        d = tokens[model].iloc[::-1]
        ax.barh(d["token"], d["score"], color="#4c72b0")
        ax.set_title(model, fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in axes[len(models):]:
        ax.axis("off")
    fig.suptitle("Most distinctive unigrams per model (one-vs-rest chi2)", y=1.001)
    fig.tight_layout()
    return _finish(fig, axes, save_path)


def plot_log_odds_fingerprints(
    profiles: dict[str, pd.DataFrame],
    models: list[str] | None = None,
    save_path: str | Path | None = None,
):
    """Grid of per-model verbal fingerprints (log-odds z-score).

    Each panel shows the words that model uses more than the other
    seventeen, ranked by effect size.  A long, confident bar list means a
    real habit; a short one means the model has no lexical tell.
    """
    import matplotlib.pyplot as plt

    models = models or list(profiles.keys())
    ncols = 3
    nrows = -(-len(models) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, model in zip(axes, models, strict=False):
        d = profiles[model]
        if d.empty:
            ax.axis("off")
            continue
        d = d.iloc[::-1]
        colours = _provider_colors([model]) * len(d)
        ax.barh(d["word"], d["z"], color=colours)
        ax.axvline(0, color="#999999", linewidth=0.8)
        ax.set_title(model, fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in axes[len(models):]:
        ax.axis("off")
    fig.suptitle(
        "Verbal fingerprint per model — log-odds z-score vs the other 17",
        y=1.001,
    )
    fig.tight_layout()
    return _finish(fig, axes, save_path)


def plot_family_fingerprints(
    profiles: dict[str, pd.DataFrame], save_path: str | Path | None = None
):
    """One panel per lab: the words that whole family favours."""
    import matplotlib.pyplot as plt

    families = list(profiles.keys())
    ncols = 3
    nrows = -(-len(families) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, family in zip(axes, families, strict=False):
        d = profiles[family]
        if d.empty:
            ax.axis("off")
            continue
        d = d.iloc[::-1]
        ax.barh(d["word"], d["z"], color="#4c72b0")
        ax.axvline(0, color="#999999", linewidth=0.8)
        ax.set_title(family, fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in axes[len(families):]:
        ax.axis("off")
    fig.suptitle("Register per lab — log-odds z-score vs the other labs", y=1.001)
    fig.tight_layout()
    return _finish(fig, axes, save_path)


def plot_group_sizes(
    sizes: np.ndarray,
    expected: int | None = None,
    save_path: str | Path | None = None,
):
    """Histogram of recovered group sizes.

    The dashed line marks the design size (18 passages per prompt): if the
    recovered groups cluster around it, the transductive grouping has found
    the prompt structure.
    """
    import matplotlib.pyplot as plt

    sizes = np.asarray(sizes)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.arange(0.5, max(sizes.max() + 1.5, 20.5), 1.0)
    ax.hist(sizes, bins=bins, color="#4c72b0", edgecolor="white")
    if expected is not None:
        ax.axvline(expected + 0.5, color="#c44e52", linestyle="--", linewidth=1.4,
                   label=f"design size ({expected} passages per prompt)")
        ax.legend(fontsize=9)
    ax.set_xlabel("passages in recovered group")
    ax.set_ylabel("number of groups")
    ax.set_title("Recovered prompt groups: size distribution")
    return _finish(fig, ax, save_path)


def plot_ablation(
    steps: list[tuple[str, float]],
    baseline: float | None = None,
    save_path: str | Path | None = None,
):
    """Waterfall chart of what each pipeline layer is worth.

    *steps* is ``[(label, balanced_accuracy), ...]`` in the order the layers
    are applied; the bars show the running score and the annotations the
    gain of each layer.
    """
    import matplotlib.pyplot as plt

    labels = [s[0] for s in steps]
    values = [s[1] for s in steps]
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(range(len(values)), values, color="#4c72b0", width=0.62)
    for i, value in enumerate(values):
        ax.text(i, value + 0.006, f"{value:.3f}", ha="center", fontsize=9)
        if i > 0:
            delta = value - values[i - 1]
            ax.text(i, value * 0.5, f"{delta:+.3f}", ha="center", fontsize=9,
                    color="#c44e52" if delta < 0 else "#2a7f62", fontweight="bold")
    if baseline is not None:
        ax.axhline(baseline, color="#999999", linestyle=":", linewidth=1.2,
                   label=f"chance ({baseline:.3f})")
        ax.legend(fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("group-CV balanced accuracy")
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_title("What each layer of the stack is worth")
    return _finish(fig, ax, save_path)


def plot_view_examples(
    examples: dict[str, str], save_path: str | Path | None = None
):
    """Show what each linguistic view does to the same passage."""
    import textwrap

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(examples), 1, figsize=(13, 2.1 * len(examples))
    )
    axes = np.atleast_1d(axes)
    for ax, (name, text) in zip(axes, examples.items(), strict=False):
        ax.axis("off")
        ax.set_title(name, loc="left", fontsize=10, fontweight="bold")
        ax.text(0, 0.9, textwrap.fill(text, 150), fontsize=8, va="top", family="monospace")
    fig.suptitle("One passage, five views", y=1.0)
    fig.tight_layout()
    return _finish(fig, axes, save_path)

"""Central configuration: paths, labels, constants.

Paths resolve transparently between local dev and the Kaggle code-competition
runtime:

* locally:    ``data/raw`` inside the repo (override with ``NTL_DATA_DIR``)
* on Kaggle:  discovered by :func:`competition_dirs`, which searches the
  whole ``/kaggle/input`` mount rather than assuming one layout
"""

from __future__ import annotations

import os
from pathlib import Path

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _find_project_root() -> Path:
    """Best-effort project root, robust to being exec'd from an inlined cell.

    Searches for the nearest ancestor of ``__file__`` (when present) or the
    current working directory that contains a ``pyproject.toml``; falls back
    to the cwd.
    """
    starts: list[Path] = []
    try:
        starts.append(Path(__file__).resolve().parent)
    except NameError:  # inlined cell: no __file__
        pass
    starts.append(Path.cwd())
    for start in starts:
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").exists():
                return candidate
    return Path.cwd()


PROJECT_ROOT = _find_project_root()
DATA_DIR = Path(os.environ.get("NTL_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
#: Root of everything Kaggle mounts for a notebook.
KAGGLE_INPUT = Path("/kaggle/input")

#: Name of the competition directory inside the Kaggle mount.
COMPETITION_NAME = "name-that-llm"

OUTPUT_DIR = PROJECT_ROOT / "outputs"


#: Files that identify the competition's own data directory.
COMPETITION_FILES = ("train.csv", "test.csv", "sample_submission.parquet")


def _dir_data_score(path: Path) -> int:
    """How many of :data:`COMPETITION_FILES` live directly in *path*."""
    return sum(1 for f in COMPETITION_FILES if (path / f).exists())


def _walk_for_data_dirs(root: Path, max_depth: int = 5) -> list[Path]:
    """Every directory under *root* that directly holds a competition file."""
    found: list[Path] = []
    root = Path(root)
    if not root.exists():
        return found
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if _dir_data_score(current) > 0:
            found.append(current)
        if depth >= max_depth:
            continue
        try:
            children = sorted(p for p in current.iterdir() if p.is_dir())
        except OSError:
            continue
        stack.extend((child, depth + 1) for child in children)
    return found


def competition_dirs() -> list[Path]:
    """Locate the competition's own data anywhere inside the Kaggle mount.

    Kaggle has used several layouts for this, and they do not all appear in
    the documentation:

    * ``/kaggle/input/competitions/name-that-llm/``
    * ``/kaggle/input/name-that-llm/``
    * ``/kaggle/input/datasets/<owner>/name-that-llm/``

    Rather than guess, we look for the directories that actually hold the
    competition's files and rank them.  The real competition directory holds
    ``train.csv`` *and* ``test.csv`` *and* ``sample_submission.parquet``; a
    public passage pool holds none of them, so the ranking cannot be fooled
    by the datasets you attach alongside the competition.

    Returns a list, best candidate first.  Empty outside the Kaggle runtime.
    """
    known = [
        KAGGLE_INPUT / "competitions" / COMPETITION_NAME,
        KAGGLE_INPUT / COMPETITION_NAME,
        KAGGLE_INPUT / "datasets" / COMPETITION_NAME,
    ]
    comps = KAGGLE_INPUT / "competitions"
    if comps.is_dir():
        known.extend(sorted(p for p in comps.iterdir() if p.is_dir()))

    candidates = [p for p in known if p.is_dir()]
    for found in _walk_for_data_dirs(KAGGLE_INPUT):
        if found not in candidates:
            candidates.append(found)

    def rank(path: Path) -> tuple:
        return (
            -_dir_data_score(path),  # most competition files first
            path.name != COMPETITION_NAME,  # then a directory named exactly so
            "competitions" not in path.parts,  # then one under competitions/
            len(path.parts),  # then the shallowest
            str(path),
        )

    return sorted(dict.fromkeys(candidates), key=rank)


def mount_summary(max_depth: int = 3, max_entries: int = 12) -> list[str]:
    """A readable dump of the Kaggle mount — used in error messages.

    Shows directories and their immediate contents, bounded in both depth
    and width so a large attached dataset cannot flood the output.
    """
    root = KAGGLE_INPUT
    if not root.exists():
        return [f"(no Kaggle mount at {root})"]
    lines: list[str] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            entries = sorted(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                lines.append("  " * depth + f"{entry.name}/")
                if depth < max_depth:
                    stack.append((entry, depth + 1))
            else:
                lines.append("  " * depth + entry.name)
        if len(entries) > max_entries:
            lines.append("  " * depth + f"... (+{len(entries) - max_entries} more)")
    return lines


def resolve_path(filename: str) -> Path:
    """Return the first existing path for *filename* across known data dirs.

    Checks ``$NTL_DATA_DIR`` (or ``data/raw``) first, then the Kaggle input
    mount. Raises a FileNotFoundError with a helpful message otherwise.
    """
    candidates = [DATA_DIR / filename]
    candidates.extend(base / filename for base in competition_dirs())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    lines = [f"Could not find {filename!r}. Tried:"]
    lines.extend(f"  {c}" for c in candidates)
    lines.append("The Kaggle mount looks like this:")
    lines.extend(f"  {line}" for line in mount_summary())
    lines.append("")
    lines.append(
        "If the competition data is not attached, open the notebook from the "
        "competition page so Kaggle mounts it, or download the data with:"
    )
    lines.append("  uv run kaggle competitions download -c name-that-llm -p data/raw")
    lines.append(
        "(requires accepting the competition rules and a kaggle.json API token)"
    )
    raise FileNotFoundError(chr(10).join(lines))


# NOTE: file paths are resolved lazily via resolve_path() in data.py so the
# package imports cleanly even before the data is downloaded.


# ---------------------------------------------------------------------------
# Labels — the 18 valid model names
# ---------------------------------------------------------------------------
LABELS: list[str] = [
    # OpenAI
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    # Google
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    # Anthropic
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    # Qwen
    "qwen3.8-max",
    # ZAI
    "glm-5.3",
    "glm-5.3-flash",
    # Moonshot AI
    "kimi-k3",
]

PROVIDERS: dict[str, str] = {
    "gpt-5.5": "OpenAI",
    "gpt-5.6-luna": "OpenAI",
    "gpt-5.6-sol": "OpenAI",
    "gpt-5.6-terra": "OpenAI",
    "gpt-6-astra": "OpenAI",
    "gemini-3.1-pro-preview": "Google",
    "gemini-3.6-flash": "Google",
    "gemini-3.7-flash": "Google",
    "gemini-3.8-flash": "Google",
    "claude-fable-5": "Anthropic",
    "claude-opus-4-6": "Anthropic",
    "claude-opus-4-8": "Anthropic",
    "claude-opus-5": "Anthropic",
    "claude-sonnet-5": "Anthropic",
    "qwen3.8-max": "Qwen",
    "glm-5.3": "ZAI",
    "glm-5.3-flash": "ZAI",
    "kimi-k3": "Moonshot AI",
}

# Column names shared across train/test
ID_COL = "id"
PROMPT_COL = "prompt_id"
TEXT_COL = "text"
TARGET_COL = "model"

# Competition description: passages are 40-250 words
MIN_WORDS, MAX_WORDS = 40, 250
N_MODELS = len(LABELS)
N_PROMPTS = 150

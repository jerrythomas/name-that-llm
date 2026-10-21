#!/usr/bin/env python
"""Inline the ``ntl`` package into a notebook as a self-contained cell.

Keeps ``src/ntl/`` as the single source of truth while making the target
notebook fully standalone (usable on Kaggle with no install and no internet).

The generated cell reconstructs the real package in ``sys.modules`` as
``ntl`` + ``ntl.<submodule>``, so notebook code keeps using
``from ntl import data, explore, ...`` unchanged.

Usage:
    uv run python scripts/sync_notebook.py
    uv run python scripts/sync_notebook.py --notebook notebooks/01_eda.ipynb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_bundle import MODULE_ORDER, read_sources  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "01_eda.ipynb"
TAG = "ntl-bundle"
MARKER = "# %% module: "
VERSION = "0.3.0"  # keep in sync with pyproject.toml

CELL_TEMPLATE = '''# ==== ntl package (INLINED — auto-generated, do not edit) ====
# Source of truth: src/ntl/*.py
# Regenerate with:  uv run python scripts/sync_notebook.py
# Makes this notebook self-contained for Kaggle (no pip install, no internet).
import sys as _sys
import types as _types

_NTL_MARKER = "__MARKER__"

_NTL_SOURCES = r\'\'\'
__BLOB__
\'\'\'


def _ntl_bootstrap():
    """Materialise the `ntl` package and its submodules from inline source."""
    chunks = _NTL_SOURCES.split("\\n" + _NTL_MARKER)[1:]
    sources = {}
    for chunk in chunks:
        name, _, body = chunk.partition("\\n")
        sources[name.strip()] = body

    pkg = _types.ModuleType("ntl")
    pkg.__path__ = []
    pkg.__version__ = "__VERSION__"
    _sys.modules["ntl"] = pkg

    order = __ORDER__
    for name in order:
        mod = _types.ModuleType("ntl." + name)
        mod.__package__ = "ntl"
        _sys.modules["ntl." + name] = mod
        setattr(pkg, name, mod)
    for name in order:
        exec(compile(sources[name], "ntl/" + name + ".py", "exec"),
             _sys.modules["ntl." + name].__dict__)


_ntl_bootstrap()
'''


def build_cell_source() -> str:
    sources = read_sources()

    for stem, src in sources.items():
        if MARKER in src:
            raise SystemExit(f"{stem!r} contains the reserved marker {MARKER!r}")
        if "'''" in src:
            raise SystemExit(
                f"{stem!r} contains a triple single-quote; cannot embed in a raw literal"
            )

    blob = "\n" + "\n".join(
        f"{MARKER}{stem}\n{src.rstrip()}" for stem, src in sources.items()
    )
    order = [name.removesuffix(".py") for name in MODULE_ORDER]

    return (
        CELL_TEMPLATE.replace("__MARKER__", MARKER)
        .replace("__ORDER__", repr(order))
        .replace("__VERSION__", VERSION)
        .replace("__BLOB__", blob)
    )


def sync(notebook_path: Path) -> None:
    cell_source = build_cell_source()
    lines = cell_source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    nb = json.loads(notebook_path.read_text())
    cells = nb["cells"]
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "ntl-bundle",
        "metadata": {"tags": [TAG]},
        "outputs": [],
        "source": lines,
    }

    idx = next(
        (i for i, c in enumerate(cells) if TAG in c.get("metadata", {}).get("tags", [])),
        None,
    )
    if idx is None:
        # Insert immediately before the first code cell so imports resolve.
        insert_at = next(
            (i for i, c in enumerate(cells) if c["cell_type"] == "code"), len(cells)
        )
        cells.insert(insert_at, new_cell)
        action = f"inserted into {notebook_path.name} at cell {insert_at}"
    else:
        # Keep the cell id so the notebook stays a valid nbformat 4.5 document.
        new_cell["id"] = cells[idx].get("id", new_cell.get("id", "ntl-bundle"))
        cells[idx] = new_cell
        action = f"updated cell {idx} in {notebook_path.name}"

    notebook_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"Synced ntl package → {action} ({len(lines)} lines)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    args = parser.parse_args()
    sync(args.notebook)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

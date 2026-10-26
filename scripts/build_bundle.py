#!/usr/bin/env python
"""Generate offline bundles of the ``ntl`` package.

Kaggle code-competition reruns have no internet and no ``uv``/editable
install, so ``import ntl`` fails unless the code is shipped another way.

Two outputs, regenerated from the modular source (single source of truth):

  * ``dist/ntl_bundle.py`` — a flat, single-file bundle. Usable via
    ``%run`` / ``sys.path`` from an attached Kaggle Dataset, or pasted as a cell.
  * the notebook's inline bootstrap cell (see ``sync_notebook.py``), which
    materialises the real ``ntl`` package in-memory so notebook code keeps
    using ``from ntl import ...`` unchanged.

Usage:
    uv run python scripts/build_bundle.py
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ntl"
OUT = ROOT / "dist" / "ntl_bundle.py"

# Dependency order — config first, then everything that imports it.
MODULE_ORDER = [
    "config.py",
    "data.py",
    "features.py",
    "views.py",
    "facets.py",
    "stylometry.py",
    "groups.py",
    "decision.py",
    "pipeline.py",
    "measure.py",
    "tracker.py",
    "explore.py",
    "plots.py",
    "baseline.py",
    "submit.py",
]

HEADER = '''"""ntl_bundle — auto-generated single-file bundle of the ``ntl`` package.

DO NOT EDIT BY HAND. Regenerate with:  uv run python scripts/build_bundle.py
Source of truth: src/ntl/*.py
"""

from __future__ import annotations
'''


def read_sources() -> dict[str, str]:
    """Return ``{module_stem: source}`` for the package, in dependency order."""
    sources: dict[str, str] = {}
    for name in MODULE_ORDER:
        sources[name.removesuffix(".py")] = (SRC / name).read_text()
    return sources


def strip_relative_imports(source: str) -> str:
    """Remove ``from .x import ...`` and ``from __future__ import ...`` lines.

    Uses the AST so multi-line parenthesised imports are handled correctly.
    (Only needed for the flat bundle; the notebook bootstrap keeps real imports.)
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    drop: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if (node.level and node.level > 0) or node.module == "__future__":
                drop.update(range(node.lineno - 1, node.end_lineno))
    return "\n".join(line for i, line in enumerate(lines) if i not in drop)


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"source package not found: {SRC}")

    parts = [HEADER]
    for stem, src in read_sources().items():
        body = strip_relative_imports(src)
        parts.append(f"\n\n# {'=' * 72}\n# {stem}.py\n# {'=' * 72}\n{body.strip()}\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(parts))
    print(f"Wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

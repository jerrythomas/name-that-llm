#!/usr/bin/env python
"""Verify that the committed notebooks match the sources they are built from.

Three invariants, all cheap:

1. ``notebooks/02_style_and_structure.ipynb`` contains exactly the cell
   *sources* that ``scripts/make_notebook.py`` generates.  Outputs and
   execution counts are ignored — they are runtime artefacts, not source.
2. Every notebook carrying the ``ntl-bundle`` tag embeds exactly the current
   ``src/ntl`` sources, so the inline copy cannot drift from the package.
3. The embedded package really imports and exposes the expected submodules.

Together these keep the repository honest: the notebook builder is the
single source of truth for notebook 02, ``src/ntl`` is the single source of
truth for the package, and the two are provably in sync.

Usage:
    uv run python scripts/check_notebooks.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_notebook  # noqa: E402
import sync_notebook  # noqa: E402

BUNDLE_TAG = sync_notebook.TAG
NOTEBOOKS = [
    ROOT / "notebooks" / "01_eda.ipynb",
    ROOT / "notebooks" / "02_style_and_structure.ipynb",
]
GENERATED = ROOT / "notebooks" / "02_style_and_structure.ipynb"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _source(cell: dict) -> str:
    src = cell.get("source", [])
    return src if isinstance(src, str) else "".join(src)


def _bundle_index(cells: list[dict]) -> int | None:
    for i, cell in enumerate(cells):
        if BUNDLE_TAG in cell.get("metadata", {}).get("tags", []):
            return i
    return None


def check_generated_sources() -> list[str]:
    """Notebook 02 must contain exactly the cells make_notebook.py generates."""
    problems: list[str] = []
    if not GENERATED.exists():
        return [f"{GENERATED.relative_to(ROOT)} is missing"]

    cells = _load(GENERATED)["cells"]
    idx = _bundle_index(cells)
    without_bundle = cells if idx is None else cells[:idx] + cells[idx + 1 :]
    expected = make_notebook.CELLS

    if len(without_bundle) != len(expected):
        problems.append(
            f"notebook 02 has {len(without_bundle)} cells "
            f"(excluding the bundle) but the builder generates {len(expected)}"
        )
        return problems

    for i, (got, want) in enumerate(zip(without_bundle, expected, strict=True)):
        if got.get("cell_type") != want["cell_type"]:
            problems.append(
                f"cell {i}: type {got.get('cell_type')!r} != {want['cell_type']!r}"
            )
            continue
        if _source(got) != _source(want):
            problems.append(f"cell {i}: source differs from scripts/make_notebook.py")
    return problems


def check_bundle_sync() -> list[str]:
    """Each notebook's inline package must equal the current src/ntl."""
    problems: list[str] = []
    expected = sync_notebook.build_cell_source().rstrip("\n")
    for path in NOTEBOOKS:
        if not path.exists():
            problems.append(f"{path.relative_to(ROOT)} is missing")
            continue
        cells = _load(path)["cells"]
        idx = _bundle_index(cells)
        if idx is None:
            problems.append(f"{path.name}: no cell tagged {BUNDLE_TAG!r}")
            continue
        got = _source(cells[idx]).rstrip("\n")
        if got != expected:
            problems.append(
                f"{path.name}: the inline package is out of date — "
                "run scripts/sync_notebook.py"
            )
    return problems


def check_bundle_imports() -> list[str]:
    """The inline package must actually import in a clean interpreter."""
    probe = (
        sync_notebook.build_cell_source()
        + "\nimport ntl\n"
        + "import ntl.config, ntl.data, ntl.views, ntl.groups\n"
        + "import ntl.decision, ntl.pipeline, ntl.explore, ntl.plots\n"
        + "import ntl.baseline, ntl.stylometry, ntl.features, ntl.submit\n"
        + "print('NTL_BUNDLE_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )
    if proc.returncode != 0 or "NTL_BUNDLE_OK" not in proc.stdout:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        detail = tail[-1] if tail else f"exit code {proc.returncode}"
        return [f"the inline package does not import cleanly: {detail}"]
    return []


def main() -> int:
    checks = [
        ("notebook 02 matches scripts/make_notebook.py", check_generated_sources),
        ("inline packages match src/ntl", check_bundle_sync),
        ("inline package imports cleanly", check_bundle_imports),
    ]
    failed = 0
    for name, check in checks:
        problems = check()
        if problems:
            failed += 1
            print(f"FAIL  {name}")
            for problem in problems:
                print(f"        {problem}")
        else:
            print(f"ok    {name}")
    if failed:
        print(f"\n{failed} check(s) failed")
        return 1
    print("\nall notebook checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

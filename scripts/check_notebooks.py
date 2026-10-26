#!/usr/bin/env python
"""Verify that the committed notebooks match the sources they are built from.

Three invariants, all cheap:

1. Every generated notebook contains exactly the cell *sources* that its
   builder (``make_notebook``, ``make_facets_notebook``, ``make_variants``)
   generates.  Outputs and execution counts are ignored — they are runtime
   artefacts, not source.
2. Every notebook carrying the ``ntl-bundle`` tag embeds exactly the current
   ``src/ntl`` sources, so the inline copy cannot drift from the package.
3. The embedded package really imports and exposes the expected submodules.

Together these keep the repository honest: the builders are the single source
of truth for the notebooks, ``src/ntl`` is the single source of truth for the
package, and the two are provably in sync.

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

import make_facets_notebook  # noqa: E402
import make_notebook  # noqa: E402
import make_variants  # noqa: E402
import sync_notebook  # noqa: E402

BUNDLE_TAG = sync_notebook.TAG
NOTEBOOKS = [
    ROOT / "notebooks" / "01_eda.ipynb",
    ROOT / "notebooks" / "02_style_and_structure.ipynb",
    ROOT / "notebooks" / "03_facets.ipynb",
    *[make_variants.path_for(variant) for variant in make_variants.VARIANTS],
]

# (notebook, callable returning the expected cells, builder path)
GENERATED: list[tuple[Path, object, str]] = [
    (
        ROOT / "notebooks" / "02_style_and_structure.ipynb",
        lambda: make_notebook.CELLS,
        "scripts/make_notebook.py",
    ),
    (
        ROOT / "notebooks" / "03_facets.ipynb",
        lambda: make_facets_notebook.CELLS,
        "scripts/make_facets_notebook.py",
    ),
    *[
        (
            make_variants.path_for(variant),
            (lambda variant=variant: make_variants.cells_for(variant)),
            "scripts/make_variants.py",
        )
        for variant in make_variants.VARIANTS
    ],
]


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
    """Every generated notebook must match its builder cell for cell."""
    problems: list[str] = []
    for path, expected_cells, builder in GENERATED:
        name = path.relative_to(ROOT)
        if not path.exists():
            problems.append(f"{name} is missing")
            continue

        cells = _load(path)["cells"]
        idx = _bundle_index(cells)
        without_bundle = cells if idx is None else cells[:idx] + cells[idx + 1 :]
        expected = expected_cells()

        if len(without_bundle) != len(expected):
            problems.append(
                f"{name} has {len(without_bundle)} cells (excluding the bundle) "
                f"but {builder} generates {len(expected)}"
            )
            continue

        for i, (got, want) in enumerate(zip(without_bundle, expected, strict=True)):
            if got.get("cell_type") != want["cell_type"]:
                problems.append(f"{name} cell {i}: type mismatch")
                continue
            if _source(got) != _source(want):
                problems.append(
                    f"{name} cell {i}: source differs ({builder}); fix with: make notebook"
                )
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
                f"{path.name}: the inline package is out of date — fix with: make notebook"
            )
    return problems


def check_bundle_imports() -> list[str]:
    """The inline package must actually import in a clean interpreter."""
    modules = [
        "config",
        "data",
        "features",
        "views",
        "facets",
        "groups",
        "decision",
        "pipeline",
        "measure",
        "tracker",
        "explore",
        "plots",
        "baseline",
        "stylometry",
        "submit",
    ]
    probe = (
        sync_notebook.build_cell_source()
        + "\nimport ntl\n"
        + "".join(f"import ntl.{name}\n" for name in modules)
        + "print('NTL_BUNDLE_OK')\n"
    )
    # Feed the probe on stdin, not with ``-c``: the bundle is ~130 KB, and
    # Linux caps a single argv entry at 128 KB (MAX_ARG_STRLEN).  ``-c`` works
    # on macOS and fails on the CI runner with E2BIG.
    proc = subprocess.run(
        [sys.executable, "-"],
        input=probe,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0 or "NTL_BUNDLE_OK" not in proc.stdout:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        detail = tail[-1] if tail else f"exit code {proc.returncode}"
        return [f"the inline package does not import cleanly: {detail}"]
    return []


def main() -> int:
    checks = [
        ("generated notebooks match their builders", check_generated_sources),
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

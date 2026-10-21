#!/usr/bin/env python
"""Execute a notebook in place with nbclient.

The notebooks in this repo ship their outputs, so re-running one locally is
how the outputs are refreshed after a change to ``src/ntl`` or to the
notebook builder.

Usage:
    uv run python scripts/run_notebook.py notebooks/02_style_and_structure.ipynb
    uv run python scripts/run_notebook.py notebooks/01_eda.ipynb --save

Without ``--save`` the executed notebook is discarded (useful as a smoke
test: the run fails loudly on the first broken cell).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def run(path: Path, save: bool, timeout: int) -> int:
    if not path.exists():
        print(f"no such notebook: {path}", file=sys.stderr)
        return 2

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(path.parent)}},
    )
    started = time.monotonic()
    client.execute()
    elapsed = time.monotonic() - started
    print(f"OK  {path.name}  ({elapsed:.0f}s, {len(nb.cells)} cells)")

    if save:
        nbformat.write(nb, path)
        print(f"saved executed notebook -> {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--save", action="store_true", help="write the outputs back")
    parser.add_argument("--timeout", type=int, default=3600, help="per-cell timeout (s)")
    args = parser.parse_args()
    return run(args.notebook, args.save, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())

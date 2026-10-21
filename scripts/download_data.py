#!/usr/bin/env python
"""Download and unzip the Name That LLM competition data into data/raw.

Prerequisites:
  1. Accept the competition rules on the Kaggle website.
  2. Provide API credentials at ~/.kaggle/kaggle.json (or set
     KAGGLE_USERNAME / KAGGLE_KEY).

Usage:
    uv run python scripts/download_data.py
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "name-that-llm"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {COMPETITION} -> {DEST}")
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "competitions", "download",
            "-c", COMPETITION,
            "-p", str(DEST),
        ],
        check=False,
    )
    if result.returncode != 0:
        print(
            "\nDownload failed. Check that you accepted the rules at\n"
            f"  https://www.kaggle.com/competitions/{COMPETITION}/rules\n"
            "and that ~/.kaggle/kaggle.json is present.",
            file=sys.stderr,
        )
        return result.returncode

    for zpath in DEST.glob("*.zip"):
        print(f"Unzipping {zpath.name}")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(DEST)
        zpath.unlink()

    print("Contents:")
    for p in sorted(DEST.glob("*")):
        print(" -", p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

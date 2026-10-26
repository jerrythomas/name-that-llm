#!/usr/bin/env python
"""Record and compare variant scores: local grouped CV vs public leaderboard.

The notebooks print a ``TRACKER {...}`` JSON line and write
``/kaggle/working/tracker.csv``.  Either paste the line with ``add`` or merge
the downloaded CSV, then read the table with ``report``.

Usage:
    uv run python scripts/track.py report
    uv run python scripts/track.py add --variant v5 --local-cv 0.4897 --public-lb 0.35
    uv run python scripts/track.py merge ~/Downloads/tracker.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ntl import tracker  # noqa: E402

DEFAULT = Path("metrics/tracker.csv")


def _split(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def cmd_add(args: argparse.Namespace) -> int:
    row = tracker.make_row(
        variant=args.variant,
        views=_split(args.views),
        facets=_split(args.facets),
        facet_weight=args.facet_weight,
        local_cv=args.local_cv,
        local_folds=args.local_folds,
        public_lb=args.public_lb,
        n_train=args.n_train,
        n_pool=args.n_pool,
        notes=args.notes,
    )
    path = tracker.append_csv(args.path, row)
    print(f"appended to {path}")
    print(tracker.to_json(row))
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    added = tracker.merge(args.path, args.source)
    print(f"merged {added} new row(s) from {args.source} into {args.path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    frame = tracker.report(args.path)
    if frame.empty:
        print(f"no rows in {args.path} yet")
        return 0
    columns = [
        "variant",
        "config_hash",
        "local_cv",
        "public_lb",
        "gap",
        "n_train",
        "n_pool",
        "views",
        "facets",
        "run_at",
    ]
    print(frame[columns].to_string(index=False))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    frame = tracker.load(args.path)
    if frame.empty:
        print(f"no rows in {args.path} yet")
        return 0
    print(frame.to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", default=str(DEFAULT), help=f"tracker CSV (default {DEFAULT})")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="record one run")
    add.add_argument("--variant", required=True)
    add.add_argument("--views", default="char,word,funcon")
    add.add_argument("--facets", default="")
    add.add_argument("--facet-weight", type=float, default=1.0)
    add.add_argument("--local-cv", type=float, default=None)
    add.add_argument("--local-folds", type=int, default=None)
    add.add_argument("--public-lb", type=float, default=None)
    add.add_argument("--n-train", type=int, default=0)
    add.add_argument("--n-pool", type=int, default=0)
    add.add_argument("--notes", default="")
    add.set_defaults(func=cmd_add)

    merge = sub.add_parser("merge", help="append rows from another tracker CSV")
    merge.add_argument("source")
    merge.set_defaults(func=cmd_merge)

    report = sub.add_parser("report", help="latest score per variant")
    report.set_defaults(func=cmd_report)

    listing = sub.add_parser("list", help="every recorded run")
    listing.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

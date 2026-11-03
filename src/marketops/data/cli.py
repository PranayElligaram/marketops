"""CLI: `python -m marketops.data.cli ingest|status`."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from marketops.data.provider import ProviderError, YahooProvider
from marketops.data.store import (
    BENCHMARK,
    DEFAULT_UNIVERSE,
    ingest_symbol,
    load_symbol,
    write_manifest,
)

DEFAULT_ROOT = Path("data/raw")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="marketops-data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ingest", "status"):
        sp = sub.add_parser(name)
        sp.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE + [BENCHMARK])
        sp.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        if name == "ingest":
            sp.add_argument("--start", type=date.fromisoformat, default=date(2005, 1, 1))
            sp.add_argument(
                "--end", type=date.fromisoformat, default=date.today() + timedelta(days=1)
            )
    args = ap.parse_args(argv)

    frames: dict[str, pd.DataFrame] = {}
    failed = 0
    provider = YahooProvider() if args.cmd == "ingest" else None
    for sym in (s.upper() for s in args.symbols):
        try:
            if provider:
                frame, added = ingest_symbol(provider, args.root, sym, args.start, args.end)
                print(f"{sym}: {len(frame)} rows (+{added})")
            else:
                frame = load_symbol(args.root, sym)
                first, last = frame["date"].min().date(), frame["date"].max().date()
                print(f"{sym}: {len(frame)} rows {first}..{last}")
            frames[sym] = frame
        except (ProviderError, ValueError, FileNotFoundError) as exc:
            failed += 1
            print(f"{sym}: FAILED - {exc}", file=sys.stderr)
    if provider and frames:
        write_manifest(args.root, provider.name, frames)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

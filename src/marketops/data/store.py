"""Idempotent local cache: one CSV per symbol under data/raw/ (git-ignored), plus a manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from marketops.data.provider import Provider
from marketops.data.validation import COLUMNS, DataQualityError, clean_and_validate

DEFAULT_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "XOM", "JNJ", "PG", "KO", "WMT"]
BENCHMARK = "SPY"


def cache_path(root: Path, symbol: str) -> Path:
    return root / f"{symbol.upper()}.csv"


def load_symbol(root: Path, symbol: str) -> pd.DataFrame:
    p = cache_path(root, symbol)
    if not p.exists():
        raise FileNotFoundError(f"No cached data for {symbol}; run `marketops-data ingest`")
    return clean_and_validate(pd.read_csv(p, parse_dates=["date"]), symbol)


def _check_adjustment_consistent(
    old: pd.DataFrame, fresh: pd.DataFrame, symbol: str, rtol: float = 1e-4
) -> None:
    """Refuse to splice fresh rows onto cached history adjusted under a different basis.

    `adj_close` is retroactively rescaled by the provider whenever a dividend is paid. If the
    overlapping dates disagree but cached rows outside the fresh window would be kept, the merged
    series would contain a fake return at the seam, so a full-history re-ingest is required.
    """
    overlap = old.merge(fresh, on="date", suffixes=("_old", "_new"))
    if overlap.empty:
        return
    rel = (overlap["adj_close_new"] / overlap["adj_close_old"] - 1).abs()
    kept_outside = (~old["date"].isin(fresh["date"])).any()
    if kept_outside and (rel > rtol).any():
        raise DataQualityError(
            f"{symbol}: provider adjusted prices changed on overlapping dates; "
            "re-ingest the full history (--start 2005-01-01) instead of splicing"
        )


def ingest_symbol(
    provider: Provider, root: Path, symbol: str, start: date, end: date
) -> tuple[pd.DataFrame, int]:
    """Fetch, validate, merge into cache (fresh rows replace overlapping dates).

    Returns (full frame, rows_added). Re-running with identical provider data adds nothing.
    """
    fresh = clean_and_validate(provider.fetch_daily(symbol, start, end), symbol)
    p = cache_path(root, symbol)
    n_old = 0
    if p.exists():
        old = clean_and_validate(pd.read_csv(p, parse_dates=["date"]), symbol)
        n_old = len(old)
        _check_adjustment_consistent(old, fresh, symbol)
        fresh = pd.concat([old[~old["date"].isin(fresh["date"])], fresh])
        fresh = fresh.sort_values("date").reset_index(drop=True)
    merged = clean_and_validate(fresh, symbol)
    root.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    merged[COLUMNS].to_csv(tmp, index=False, date_format="%Y-%m-%d")
    tmp.replace(p)
    return merged, len(merged) - n_old


def fingerprint(root: Path, symbols: list[str]) -> dict[str, dict[str, object]]:
    """SHA-256, row count and date range of each cached CSV (as validated)."""
    out: dict[str, dict[str, object]] = {}
    for s in symbols:
        f = load_symbol(root, s)
        out[s] = {
            "sha256": hashlib.sha256(cache_path(root, s).read_bytes()).hexdigest(),
            "rows": len(f),
            "first": str(f["date"].min().date()),
            "last": str(f["date"].max().date()),
        }
    return out


def write_manifest(root: Path, provider_name: str, frames: dict[str, pd.DataFrame]) -> Path:
    """Record ingest metadata; merges with an existing manifest so a partial re-ingest of a few
    symbols cannot erase provenance of the others."""
    p = root / "manifest.json"
    previous: dict[str, dict[str, object]] = {}
    if p.exists():
        try:
            previous = json.loads(p.read_text()).get("symbols", {})
        except ValueError:
            previous = {}
    manifest = {
        "provider": provider_name,
        "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "terms": "Personal research use only; not redistributable. Do not commit.",
        "symbols": previous
        | {
            s: {
                "rows": len(f),
                "first": str(f["date"].min().date()),
                "last": str(f["date"].max().date()),
            }
            for s, f in frames.items()
        },
    }
    p.write_text(json.dumps(manifest, indent=2))
    return p

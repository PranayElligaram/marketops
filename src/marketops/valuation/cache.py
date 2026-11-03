"""Local, git-ignored cache of provider daily bars (one CSV + one metadata file per symbol).

Data comes from the same personal-use Yahoo chart provider as the research pipeline and is never
redistributed. Only the public symbol and date range are sent to the provider.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from marketops.data.provider import Provider, ProviderError
from marketops.valuation.nav import PriceSeries

_SAFE = re.compile(r"^[A-Za-z0-9.\-^=]{1,24}$")
COLUMNS = ["date", "open", "high", "low", "close", "volume", "dividend", "split"]


def _paths(root: Path, symbol: str) -> tuple[Path, Path]:
    if not _SAFE.match(symbol):
        raise ValueError("unsafe symbol")
    return root / f"{symbol}.csv", root / f"{symbol}.meta.json"


def load_series(root: Path, symbol: str) -> PriceSeries | None:
    csv, meta = _paths(root, symbol)
    if not csv.exists() or not meta.exists():
        return None
    frame = pd.read_csv(csv, parse_dates=["date"])
    if not set(COLUMNS) <= set(frame.columns):
        return None
    info = json.loads(meta.read_text(encoding="utf-8"))
    frame = frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    # The provider returns null bars on some exchange holidays: a missing bar is missing, not zero.
    ok = frame[["close", "low", "high"]].gt(0).all(axis=1)
    frame = frame[ok].reset_index(drop=True)
    return PriceSeries(symbol, info.get("currency"), frame[COLUMNS])


def store_series(root: Path, symbol: str, frame: pd.DataFrame, currency: str | None) -> None:
    csv, meta = _paths(root, symbol)
    root.mkdir(parents=True, exist_ok=True)
    frame[COLUMNS].to_csv(csv, index=False)
    meta.write_text(
        json.dumps(
            {
                "currency": currency,
                "retrieved_utc": datetime.now(UTC).isoformat(),
                "note": "Personal research use only; not redistributable.",
            }
        ),
        encoding="utf-8",
    )


def fetch_symbol(
    root: Path, provider: Provider, symbol: str, first_needed: date, today: date | None = None
) -> str:
    """Download and cache one symbol from slightly before `first_needed`. Returns a status."""
    end = (today or datetime.now(UTC).date()) + timedelta(days=1)
    try:
        frame = provider.fetch_daily(symbol, first_needed - timedelta(days=14), end)
    except ProviderError as exc:
        return f"failed: {exc}"
    currency = frame.attrs.get("currency")
    store_series(root, symbol, frame, currency)
    return f"ok: {len(frame)} rows"

"""SPY total-return benchmark data: local cache, provenance and conversion to account currency.

Data comes from the same personal-use provider as the rest of the project, is cached under the
git-ignored market cache and is never redistributed. Only the public symbol and date range are
sent to the provider. `adj_close` (split- and dividend-adjusted) is the total-return proxy: it is
used here and nowhere else, because the portfolio valuation deliberately avoids adjusted prices.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd

from marketops.data.provider import Provider, ProviderError
from marketops.valuation.benchmark import BenchmarkPrice
from marketops.valuation.comparison import BenchmarkInput, BenchmarkSource
from marketops.valuation.fx import FxTable

SYMBOL = "SPY"
DIRNAME = "benchmarks"
COLUMNS = ["date", "close", "adj_close", "dividend"]
BASIS = "Dividend-adjusted close (total-return proxy), USD, converted with ECB reference rates."


def _paths(root: Path) -> tuple[Path, Path]:
    d = root / DIRNAME
    return d / f"{SYMBOL}.csv", d / f"{SYMBOL}.meta.json"


def fetch_benchmark(
    root: Path, provider: Provider, first_needed: date, today: date | None = None
) -> str:
    """Download and cache SPY from slightly before `first_needed`. Returns a status string."""
    end = (today or datetime.now(UTC).date()) + timedelta(days=1)
    try:
        frame = provider.fetch_daily(SYMBOL, first_needed - timedelta(days=14), end)
    except ProviderError as exc:
        return f"failed: {exc}"
    if "adj_close" not in frame.columns:
        return "failed: provider response has no adjusted close"
    csv, meta = _paths(root)
    csv.parent.mkdir(parents=True, exist_ok=True)
    frame[COLUMNS].to_csv(csv, index=False)
    meta.write_text(
        json.dumps(
            {
                "symbol": SYMBOL,
                "provider": getattr(provider, "name", None),
                "currency": frame.attrs.get("currency"),
                "basis": BASIS,
                "retrieved_utc": datetime.now(UTC).isoformat(),
                "sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
                "note": "Personal research use only; not redistributable.",
            }
        ),
        encoding="utf-8",
    )
    return f"ok: {len(frame)} rows"


def to_account_currency(
    frame: pd.DataFrame, fx: FxTable, ccy: str, quote_ccy: str
) -> tuple[list[BenchmarkPrice], int, int]:
    """(prices, dropped for invalid price, dropped for missing FX). Never fills or interpolates."""
    prices: list[BenchmarkPrice] = []
    bad = no_fx = 0
    for ts, raw in zip(frame["date"], frame["adj_close"], strict=True):
        v = pd.to_numeric(raw, errors="coerce")
        if pd.isna(v) or not float("-inf") < float(v) < float("inf") or float(v) <= 0:
            bad += 1
            continue
        day = pd.Timestamp(ts).date()
        rate = fx.rate(quote_ccy, ccy, day)
        if rate is None:
            no_fx += 1
            continue
        prices.append(BenchmarkPrice(day, Decimal(str(float(v))) * rate))
    return prices, bad, no_fx


def load_benchmark(root: Path, fx: FxTable | None, ccy: str | None) -> BenchmarkInput:
    csv, meta = _paths(root)
    if not csv.exists() or not meta.exists():
        return BenchmarkInput([], None, [f"No cached {SYMBOL} benchmark data."])
    info = json.loads(meta.read_text(encoding="utf-8"))
    frame = pd.read_csv(csv, parse_dates=["date"])
    if not set(COLUMNS) <= set(frame.columns):
        return BenchmarkInput([], None, [f"Cached {SYMBOL} file has unexpected columns."])
    frame = frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    digest = hashlib.sha256(csv.read_bytes()).hexdigest()
    problems: list[str] = []
    if info.get("sha256") != digest:
        problems.append("Cached benchmark file does not match its recorded fingerprint.")
    quote = (info.get("currency") or "").upper() or None
    if quote != "USD":
        problems.append(f"Benchmark quote currency is {quote or 'unknown'}, expected USD.")
    if fx is None or not ccy:
        problems.append("Exchange rates or account currency are unavailable.")
    prices: list[BenchmarkPrice] = []
    bad = no_fx = 0
    if not problems and fx is not None and ccy:
        prices, bad, no_fx = to_account_currency(frame, fx, ccy, "USD")
    retrieved = info.get("retrieved_utc")
    age = None
    if retrieved:
        age = (datetime.now(UTC) - datetime.fromisoformat(retrieved)).days
    source = BenchmarkSource(
        symbol=SYMBOL,
        provider=info.get("provider"),
        basis=BASIS,
        quote_currency=quote,
        first_day=prices[0].day.isoformat() if prices else None,
        last_day=prices[-1].day.isoformat() if prices else None,
        rows_usable=len(prices),
        rows_dropped_invalid_price=bad,
        rows_dropped_missing_fx=no_fx,
        retrieved_utc=retrieved,
        sha256=digest,
        age_days=age,
    )
    if not problems and not prices:
        problems.append("No benchmark day has both a valid price and an exchange rate.")
    return BenchmarkInput(prices, source, problems)

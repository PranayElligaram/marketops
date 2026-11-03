"""Synthetic fixtures for valuation tests (no real data)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd

from marketops.valuation.fx import FxTable
from marketops.valuation.nav import PriceSeries


def utc(d: date, hour: int = 12) -> datetime:
    return datetime(d.year, d.month, d.day, hour, tzinfo=UTC)


def bars(
    start: date,
    closes: list[float],
    *,
    splits: dict[date, float] | None = None,
    spread: float = 0.02,
) -> pd.DataFrame:
    """Weekday bars; `closes` are the provider's (split-adjusted) closes."""
    days: list[date] = []
    d = start
    while len(days) < len(closes):
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(days),
            "open": closes,
            "high": [c * (1 + spread) for c in closes],
            "low": [c * (1 - spread) for c in closes],
            "close": closes,
            "volume": 1000.0,
            "dividend": 0.0,
            "split": [(splits or {}).get(x, 1.0) for x in days],
        }
    )


def series(symbol: str, frame: pd.DataFrame, currency: str = "USD") -> PriceSeries:
    return PriceSeries(symbol, currency, frame)


def fx_table(usd_per_eur: str = "1.25", gbp_per_eur: str = "1", days: int = 40) -> FxTable:
    start = date(2023, 12, 20)
    rows_usd, rows_gbp = {}, {}
    for i in range(days):
        d = start + timedelta(days=i)
        if d.weekday() < 5:
            rows_usd[d] = Decimal(usd_per_eur)
            rows_gbp[d] = Decimal(gbp_per_eur)
    return FxTable({"USD": rows_usd, "GBP": rows_gbp})


def fill(
    n: int,
    when: datetime,
    side: str,
    qty: str,
    value: str,
    price: float,
    ticker: str = "AAA_US_EQ",
    inst_ccy: str = "USD",
) -> dict[str, Any]:
    return {
        "order_id": n,
        "fill_id": n * 10,
        "ticker": ticker,
        "category": side,
        "occurred_at": when,
        "quantity": Decimal(qty),
        "net_value": Decimal(value),
        "price": Decimal(str(price)),
        "wallet_currency": "GBP",
        "instrument_currency": inst_ccy,
        "fx_rate": Decimal("1.25"),
        "anomaly": None,
    }


def cash_tx(n: int, when: datetime, category: str, amount: str) -> dict[str, Any]:
    return {
        "source_id": f"t{n}",
        "category": category,
        "occurred_at": when,
        "amount": Decimal(amount),
        "currency": "GBP",
        "anomaly": None,
    }


def dividend(n: int, when: datetime, amount: str) -> dict[str, Any]:
    return {
        "source_id": f"d{n}",
        "category": "dividend",
        "occurred_at": when,
        "amount": Decimal(amount),
        "currency": "GBP",
        "anomaly": None,
        "ticker": "AAA_US_EQ",
    }

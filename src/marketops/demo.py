"""Deterministic synthetic demonstration account (no credentials, no network, no real data).

Builds a complete `HistoricalValuation`, including the benchmark comparison, by running the real
valuation pipeline on invented fills, cash movements, prices, FX and a synthetic stand-in for SPY.
The figures are fictional and exist only so the dashboard and tests can show every feature.
"""

from __future__ import annotations

import math
import random
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd

from marketops.valuation.benchmark_data import BASIS, to_account_currency
from marketops.valuation.comparison import BenchmarkInput, BenchmarkSource
from marketops.valuation.fx import FxTable
from marketops.valuation.nav import PriceSeries
from marketops.valuation.report import HistoricalValuation, build_valuation

START, END = date(2023, 1, 2), date(2024, 12, 31)
AS_OF = datetime(2024, 12, 31, 18, tzinfo=UTC)
US_HOLIDAYS = {
    date(2023, 1, 2),
    date(2023, 1, 16),
    date(2023, 7, 4),
    date(2023, 11, 23),
    date(2023, 12, 25),
    date(2024, 1, 1),
    date(2024, 1, 15),
    date(2024, 7, 4),
    date(2024, 11, 28),
    date(2024, 12, 25),
}
TICKERS = {"ALPHA_US_EQ": ("ALPHA", 0.0006, 0.012), "BETA_US_EQ": ("BETA", 0.0004, 0.009)}
SYNTHETIC_NOTE = "SYNTHETIC DEMONSTRATION DATA: invented account and prices, not market data."


def _weekdays() -> list[date]:
    out, d = [], START
    while d <= END:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _walk(rng: random.Random, n: int, start: float, drift: float, vol: float) -> list[float]:
    out, p = [], start
    for _ in range(n):
        p *= 1.0 + rng.gauss(drift, vol)
        out.append(round(p, 4))
    return out


def _utc(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 14, tzinfo=UTC)


def build_demo() -> HistoricalValuation:
    rng = random.Random(20240101)
    days = _weekdays()
    usd_per_eur = {
        d: Decimal(str(round(1.10 + 0.04 * math.sin(i / 45), 5))) for i, d in enumerate(days)
    }
    gbp_per_eur = {d: Decimal("0.86") for d in days}
    fx = FxTable({"USD": usd_per_eur, "GBP": gbp_per_eur})

    def gbp_per_usd(d: date) -> Decimal:
        return gbp_per_eur[d] / usd_per_eur[d]

    closes = {t: _walk(rng, len(days), 100.0, drift, vol) for t, (_, drift, vol) in TICKERS.items()}
    frames = {
        sym: pd.DataFrame(
            {
                "date": pd.to_datetime(days),
                "open": c,
                "high": [x * 1.02 for x in c],
                "low": [x * 0.98 for x in c],
                "close": c,
                "volume": 1000.0,
                "dividend": 0.0,
                "split": 1.0,
            }
        )
        for t, (sym, _, _) in TICKERS.items()
        for c in [closes[t]]
    }
    prices = {s: PriceSeries(s, "USD", f) for s, f in frames.items()}

    # Cash movements and trades: a deposit on the first weekday of each month, bought in two names.
    tx: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    cash, qty = Decimal(0), {t: Decimal(0) for t in TICKERS}
    first_of_month = {
        min(d for d in days if (d.year, d.month) == k) for k in {(d.year, d.month) for d in days}
    }
    idx = {d: i for i, d in enumerate(days)}

    def add_fill(d: date, ticker: str, side: str, shares: Decimal) -> None:
        nonlocal cash
        price = Decimal(str(closes[ticker][idx[d]]))
        value = (shares * price * gbp_per_usd(d)).quantize(Decimal("0.01"))
        n = len(fills) + 1
        cash += -value if side == "buy" else value
        qty[ticker] += shares if side == "buy" else -shares
        fills.append(
            {
                "order_id": n, "fill_id": n * 10, "ticker": ticker, "category": side,
                "occurred_at": _utc(d), "quantity": shares, "net_value": value, "price": price,
                "wallet_currency": "GBP", "instrument_currency": "USD",
                "fx_rate": gbp_per_usd(d), "anomaly": None,
            }
        )  # fmt: skip

    for d in sorted(first_of_month):
        amount = Decimal(2000 if d == days[0] else 500)
        tx.append(
            {"source_id": f"t{len(tx)}", "category": "deposit", "occurred_at": _utc(d),
             "amount": amount, "currency": "GBP", "anomaly": None}
        )  # fmt: skip
        cash += amount
        for ticker in TICKERS:
            px = Decimal(str(closes[ticker][idx[d]])) * gbp_per_usd(d)
            add_fill(d, ticker, "buy", (amount * Decimal("0.45") / px).quantize(Decimal("0.0001")))
    sell_day = date(2024, 6, 3)
    add_fill(sell_day, "ALPHA_US_EQ", "sell", (qty["ALPHA_US_EQ"] / 4).quantize(Decimal("0.0001")))
    wd = date(2024, 9, 2)
    tx.append(
        {"source_id": "w1", "category": "withdrawal", "occurred_at": _utc(wd),
         "amount": Decimal(300), "currency": "GBP", "anomaly": None}
    )  # fmt: skip
    cash -= 300

    last = days[-1]
    values = {t: float(qty[t] * Decimal(str(closes[t][-1])) * gbp_per_usd(last)) for t in TICKERS}
    holdings = {t: float(qty[t]) for t in TICKERS}
    ledger_cash = float(cash)

    # Synthetic stand-in for SPY: priced on US trading days only; adj_close < close earlier on
    # (dividends reinvested), converging at the end like a real dividend-adjusted series.
    spy_days = [d for d in days if d not in US_HOLIDAYS]
    spy = _walk(rng, len(spy_days), 400.0, 0.0005, 0.008)
    n = len(spy_days)
    adj = [round(c * (0.97 + 0.03 * i / (n - 1)), 4) for i, c in enumerate(spy)]
    spy_frame = pd.DataFrame({"date": pd.to_datetime(spy_days), "close": spy, "adj_close": adj})
    spy_prices, bad, no_fx = to_account_currency(spy_frame, fx, "GBP", "USD")
    bench = BenchmarkInput(
        spy_prices,
        BenchmarkSource(
            symbol="SPY (synthetic)",
            provider="synthetic demonstration generator",
            basis=BASIS,
            quote_currency="USD",
            first_day=spy_prices[0].day.isoformat(),
            last_day=spy_prices[-1].day.isoformat(),
            rows_usable=len(spy_prices),
            rows_dropped_invalid_price=bad,
            rows_dropped_missing_fx=no_fx,
            retrieved_utc=AS_OF.isoformat(),
            sha256=None,
        ),
    )
    val = build_valuation(
        fills=fills,
        tx=tx,
        divs=[],
        ccy="GBP",
        as_of=AS_OF,
        price_root=None,
        fx=fx,
        overrides=None,
        history_verified=True,
        ledger_cash=ledger_cash,
        broker_positions={t: holdings[t] for t in TICKERS},
        broker_values=values,
        broker_total=ledger_cash + sum(values.values()),
        series_override=prices,
        benchmark=bench,
    )
    val.notes.insert(0, SYNTHETIC_NOTE)
    return val

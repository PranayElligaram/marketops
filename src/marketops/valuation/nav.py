"""Event-driven daily portfolio NAV in the account currency.

Conventions (all explicit, none inferred):
- Event date = the UTC calendar date of the record. A valuation day is every Monday-Friday; the
  NAV at day D is end-of-day: all events dated <= D are applied, then holdings are marked at the
  last provider close on or before D (at most `MAX_PRICE_STALE_DAYS` old, never across a split)
  and converted with the last ECB reference rate on or before D (at most `fx.MAX_STALE_DAYS` old).
- Prices are the provider's split-adjusted `close`, scaled back to the price as traded
  (x product of later splits) so they pair with share counts as held on that day. `adj_close`
  (dividend-adjusted) is never used: dividends enter only through the cash they pay, so they
  are counted exactly once.
- Provider splits multiply the quantity held on their ex-date. If Trading 212 itself reports
  corporate-action fills for an instrument, the share effect is ambiguous (the broker and the
  provider could both describe one event), so valuation of that instrument is unavailable from
  that date rather than guessed.
- A day without a defensible value for every non-zero position, or whose cash is negative
  beyond tolerance (a sign of missing history), has `nav = None` with the reasons recorded.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd

from marketops.valuation.fx import FxTable
from marketops.valuation.symbols import (
    MappingVerdict,
    as_traded_factor,
    normalise_currency,
)

MAX_PRICE_STALE_DAYS = 5
QTY_EPS = Decimal("0.000001")
NEGATIVE_CASH_TOLERANCE = Decimal("1.00")


@dataclass(frozen=True)
class PriceSeries:
    symbol: str
    currency: str | None  # as quoted by the provider (may be a minor unit such as GBp)
    frame: pd.DataFrame  # columns: date, close, low, high, split


@dataclass
class PositionValue:
    ticker: str
    quantity: Decimal
    price_date: date
    local_price: Decimal  # as traded, major currency units
    local_currency: str
    fx_rate: Decimal
    value: Decimal  # account currency
    stale_days: int


@dataclass
class DayNav:
    day: date
    nav: Decimal | None
    cash: Decimal
    holdings_value: Decimal | None
    net_invested: Decimal
    flow: Decimal  # net external flow applied since the previous valuation day
    reasons: list[str] = field(default_factory=list)
    positions: list[PositionValue] = field(default_factory=list)
    stale_price_positions: int = 0


@dataclass(frozen=True)
class _Event:
    when: datetime
    order: int
    kind: str  # deposit | withdrawal | cash | buy | sell
    ticker: str | None
    cash: Decimal
    qty: Decimal
    external: Decimal  # signed external flow (deposit +, withdrawal -)


def _d(v: Any) -> Decimal:
    return Decimal(str(v))


def _utc(v: datetime) -> datetime:
    return v.replace(tzinfo=UTC) if v.tzinfo is None else v.astimezone(UTC)


def build_events(
    tx: Sequence[Mapping[str, Any]],
    divs: Sequence[Mapping[str, Any]],
    fills: Sequence[Mapping[str, Any]],
    ccy: str,
) -> tuple[list[_Event], dict[str, date]]:
    """Cash and share events, plus the first corporate-action date per ticker.

    The cash arithmetic mirrors the reconciled cash ledger (same inclusions and signs), so the
    final cash balance is the one that was checked against the broker.
    """
    ev: list[_Event] = []
    corp: dict[str, date] = {}
    n = 0
    for r in tx:
        if r.get("anomaly") or r["currency"] != ccy:
            continue
        amt, cat = _d(r["amount"]), r["category"]
        n += 1
        if cat == "deposit":
            ev.append(
                _Event(_utc(r["occurred_at"]), n, "deposit", None, abs(amt), Decimal(0), abs(amt))
            )
        elif cat == "withdrawal":
            ev.append(
                _Event(
                    _utc(r["occurred_at"]), n, "withdrawal", None, -abs(amt), Decimal(0), -abs(amt)
                )
            )
        elif cat == "fee":
            ev.append(
                _Event(_utc(r["occurred_at"]), n, "cash", None, -abs(amt), Decimal(0), Decimal(0))
            )
        elif cat == "interest":
            ev.append(_Event(_utc(r["occurred_at"]), n, "cash", None, amt, Decimal(0), Decimal(0)))
    for r in divs:
        if r.get("anomaly") or r["currency"] != ccy:
            continue
        n += 1
        ev.append(
            _Event(_utc(r["occurred_at"]), n, "cash", None, _d(r["amount"]), Decimal(0), Decimal(0))
        )
    for r in fills:
        cat = r["category"]
        when = _utc(r["occurred_at"])
        if cat == "corporate_action":
            d = when.date()
            if r["ticker"] not in corp or d < corp[r["ticker"]]:
                corp[r["ticker"]] = d
            continue
        if cat not in ("buy", "sell") or r.get("anomaly"):
            continue
        if r.get("quantity") is None or r.get("net_value") is None:
            continue
        if r.get("wallet_currency") != ccy:
            continue
        qty, value = abs(_d(r["quantity"])), abs(_d(r["net_value"]))
        n += 1
        if cat == "buy":
            ev.append(_Event(when, n, "buy", r["ticker"], -value, qty, Decimal(0)))
        else:
            ev.append(_Event(when, n, "sell", r["ticker"], value, -qty, Decimal(0)))
    ev.sort(key=lambda e: (e.when, e.order))
    return ev, corp


def _price_on(series: PriceSeries, day: date) -> tuple[date, Decimal, int] | tuple[None, str, int]:
    """(price date, as-traded major-unit price, stale days) or (None, reason, 0)."""
    f = series.frame
    dates = pd.to_datetime(f["date"]).dt.date
    mask = dates <= day
    if not mask.any():
        return None, "no_price_yet", 0
    pos = int(mask.to_numpy().nonzero()[0][-1])
    pdate = dates.iloc[pos]
    stale = (day - pdate).days
    if stale > MAX_PRICE_STALE_DAYS:
        return None, "price_too_old", stale
    splits = f["split"].astype(float)
    # any split with ex-date in (price date, day] would make the old price a different share class
    if ((dates > pdate) & (dates <= day) & (splits != 1.0)).any():
        return None, "price_stale_across_split", stale
    factor = float(as_traded_factor(pd.Series(splits.to_numpy(), index=dates)).iloc[pos])
    close = float(f["close"].iloc[pos])
    if not (close > 0):
        return None, "non_positive_price", stale
    _, divisor = normalise_currency(series.currency)
    return pdate, Decimal(str(close * factor)) / divisor, stale


def compute_daily_nav(
    events: Sequence[_Event],
    corporate_action_dates: Mapping[str, date],
    verdicts: Mapping[str, MappingVerdict],
    prices: Mapping[str, PriceSeries],
    fx: FxTable | None,
    ccy: str,
    end: date,
) -> tuple[list[DayNav], dict[str, Decimal]]:
    """Daily NAVs and the final (split-adjusted) share counts."""
    if not events:
        return [], {}
    start = events[0].when.date()
    qty: dict[str, Decimal] = defaultdict(Decimal)
    cash = Decimal(0)
    net_invested = Decimal(0)
    flow_since = Decimal(0)
    # splits by ticker -> {ex date: ratio}
    splits: dict[str, dict[date, Decimal]] = {}
    for t, v in verdicts.items():
        s = prices.get(v.symbol or "")
        if v.status == "verified" and s is not None:
            dts = pd.to_datetime(s.frame["date"]).dt.date
            sp = s.frame["split"].astype(float)
            splits[t] = {d: Decimal(str(x)) for d, x in zip(dts, sp, strict=True) if x != 1.0}
    out: list[DayNav] = []
    i = 0
    day = start
    while day <= end:
        for t, table in splits.items():
            if day in table and qty[t] > 0 and t not in corporate_action_dates:
                qty[t] *= table[day]
        while i < len(events) and events[i].when.date() <= day:
            e = events[i]
            i += 1
            cash += e.cash
            net_invested += e.external
            flow_since += e.external
            if e.ticker is not None:
                qty[e.ticker] += e.qty
        if day.weekday() < 5:
            out.append(
                _value_day(
                    day,
                    qty,
                    cash,
                    net_invested,
                    flow_since,
                    corporate_action_dates,
                    verdicts,
                    prices,
                    fx,
                    ccy,
                )
            )
            flow_since = Decimal(0)
        day += timedelta(days=1)
    return out, {k: v for k, v in qty.items() if abs(v) > QTY_EPS}


def _value_day(
    day: date,
    qty: Mapping[str, Decimal],
    cash: Decimal,
    net_invested: Decimal,
    flow: Decimal,
    corp: Mapping[str, date],
    verdicts: Mapping[str, MappingVerdict],
    prices: Mapping[str, PriceSeries],
    fx: FxTable | None,
    ccy: str,
) -> DayNav:
    reasons: list[str] = []
    positions: list[PositionValue] = []
    total = Decimal(0)
    stale_n = 0
    for t in sorted(qty):
        q = qty[t]
        if abs(q) <= QTY_EPS:
            continue
        if q < 0:
            reasons.append(f"negative_quantity:{t}")
            continue
        if t in corp and day >= corp[t]:
            reasons.append(f"corporate_action_ambiguous:{t}")
            continue
        v = verdicts.get(t)
        if v is None or v.status != "verified":
            reasons.append(f"mapping_{v.status if v else 'unmapped'}:{t}")
            continue
        series = prices.get(v.symbol or "")
        if series is None:
            reasons.append(f"no_price_data:{t}")
            continue
        pdate, px, stale = _price_on(series, day)
        if pdate is None:
            reasons.append(f"{px}:{t}")
            continue
        assert isinstance(px, Decimal)
        major, _ = normalise_currency(series.currency)
        if major is None:
            reasons.append(f"unknown_price_currency:{t}")
            continue
        if major == ccy:
            rate: Decimal | None = Decimal(1)
        else:
            rate = fx.rate(major, ccy, day) if fx else None
        if rate is None:
            reasons.append(f"missing_fx:{major}->{ccy}:{t}")
            continue
        value = q * px * rate
        positions.append(PositionValue(t, q, pdate, px, major, rate, value, stale))
        stale_n += 1 if stale > 0 else 0
        total += value
    if cash < -NEGATIVE_CASH_TOLERANCE:
        reasons.append("negative_cash_history_incomplete")
    nav = None if reasons else total + cash
    return DayNav(
        day, nav, cash, None if reasons else total, net_invested, flow, reasons, positions, stale_n
    )

"""Contribution-matched benchmark simulation (pure; no I/O, no network, no database).

Question answered: "what would the same external deposits and withdrawals be worth if every
pound had been put into the benchmark instead?"

Inputs are already in GBP. The benchmark price series must be a *total-return* proxy (e.g.
dividend-adjusted SPY close converted to GBP on the same day), so dividends are already inside the
price and are never added separately. The simulation is invariant to a uniform rescaling of the
price series, so later back-adjustment of historical prices cannot change the output.

Conventions (match `nav.py` / `twr.py` `end_of_day`):
- A flow executes at the close of its execution day and is part of that day's value; it earns
  nothing before that close.
- Execution day = the first price day on or after the flow's own date (weekend/holiday/missing-price
  flows wait for the next priced day, exactly like flows on unvalued days in the portfolio TWR).
  Never an earlier price: no look-ahead.
- Sign: deposit +, withdrawal - (same as `DayNav.flow` and `twr.DayValue.flow`).
- No interpolation or forward-filling: the output has exactly one point per input price day.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

QTY_EPS = Decimal("0.000001")  # same tolerance as nav.py; absorbs Decimal division residue


@dataclass(frozen=True)
class BenchmarkPrice:
    day: date
    price: Decimal  # GBP per benchmark unit, total-return basis, must be finite and > 0


@dataclass(frozen=True)
class ExternalFlow:
    day: date
    amount: Decimal  # GBP; deposit +, withdrawal -; must be finite


@dataclass(frozen=True)
class BenchmarkPoint:
    day: date
    units: Decimal  # benchmark units held after this day's close (>= 0)
    value: Decimal  # units * price, GBP
    net_contributions: Decimal  # cumulative flows executed up to and including this day, GBP


def simulate_benchmark(
    prices: Sequence[BenchmarkPrice], flows: Sequence[ExternalFlow]
) -> list[BenchmarkPoint]:
    """Simulate investing `flows` in the benchmark at `prices`.

    Rules
    -----
    1. Validate (see below), then process price days in order.
    2. Net flow of a price day = sum of all flows whose execution day is that day.
    3. units_t = units_{t-1} + net_flow_t / price_t, with units_0 = 0.
    4. value_t = units_t * price_t.   net_contributions_t = sum of net flows executed <= t.
    5. If units_t < -QTY_EPS the withdrawal exceeds holdings: raise ValueError (no shorting).
       If -QTY_EPS <= units_t <= QTY_EPS set units_t to exactly Decimal(0).
    6. Return one BenchmarkPoint per input price, same order. Full Decimal precision, no rounding.

    Raises
    ------
    TypeError   a price/amount is not a Decimal (floats and ints are rejected, never coerced).
    ValueError  price days not strictly increasing (duplicate or unsorted); price non-finite or
                <= 0; flow amount non-finite; a flow dated after the last price day (cannot be
                priced; never dropped silently); a withdrawal that would make units negative
                beyond QTY_EPS; flows given but prices empty.
    Zero-amount flows are accepted and have no effect. Flows may be in any order; several on one
    date are summed. Empty prices and empty flows return [].
    """
    for p in prices:
        if type(p.day) is not date:
            raise TypeError("price day must be a datetime.date")
        if not isinstance(p.price, Decimal):
            raise TypeError("price must be a Decimal")
        if not p.price.is_finite() or p.price <= 0:
            raise ValueError(f"price on {p.day} must be finite and > 0")
    for a, b in zip(prices, prices[1:], strict=False):
        if b.day <= a.day:
            raise ValueError("price days must be strictly increasing")
    for f in flows:
        if type(f.day) is not date:
            raise TypeError("flow day must be a datetime.date")
        if not isinstance(f.amount, Decimal):
            raise TypeError("flow amount must be a Decimal")
        if not f.amount.is_finite():
            raise ValueError(f"flow amount on {f.day} must be finite")
    if not prices:
        if flows:
            raise ValueError("flows given but no prices")
        return []

    days = [p.day for p in prices]
    net = [Decimal(0)] * len(prices)
    for f in flows:
        idx = bisect_left(days, f.day)  # first priced day on or after the flow date
        if idx == len(days):
            raise ValueError(f"flow on {f.day} is after the last price day {days[-1]}")
        net[idx] += f.amount

    out: list[BenchmarkPoint] = []
    units = Decimal(0)
    contributed = Decimal(0)
    for p, flow in zip(prices, net, strict=True):
        units += flow / p.price
        contributed += flow
        if units < -QTY_EPS:
            raise ValueError(f"withdrawal on {p.day} exceeds benchmark holdings")
        if abs(units) <= QTY_EPS:
            units = Decimal(0)
        out.append(BenchmarkPoint(p.day, units, units * p.price, contributed))
    return out

"""Time-weighted return, risk and period statistics from a daily NAV series and external flows.

Convention (default `end_of_day`): an external flow on a valuation day happens after that day's
close, so the day's return is `(NAV_t - flow_t) / NAV_{t-1} - 1`; the new money earns nothing
until the next day. `start_of_day` instead assumes the flow arrives before the open:
`NAV_t / (NAV_{t-1} + flow_t) - 1`. Dividends, interest and fees are internal (they are return).

Days with no NAV (missing price/FX) break the chain: returns are linked only across consecutive
valued days, and each unbroken run is a segment. Nothing is interpolated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

FlowTiming = Literal["end_of_day", "start_of_day"]
TRADING_DAYS = 252
MIN_RETURNS_FOR_VOL = 60
MIN_RETURNS_FOR_DRAWDOWN = 20
MIN_DAYS_FOR_ANNUALISATION = 365


@dataclass(frozen=True)
class DayValue:
    day: date
    nav: float | None  # None = not valued
    flow: float  # net external flow since the previous valuation day (deposit +, withdrawal -)


@dataclass(frozen=True)
class DailyReturn:
    day: date
    value: float
    segment: int


def daily_returns(days: Sequence[DayValue], timing: FlowTiming = "end_of_day") -> list[DailyReturn]:
    out: list[DailyReturn] = []
    seg = 0
    prev: float | None = None
    gap = False
    for d in days:
        if d.nav is None:
            prev, gap = None, True
            continue
        if prev is None:
            if gap and out:
                seg += 1
            gap = False
            prev = d.nav
            continue
        if prev <= 0:
            prev = d.nav  # account unfunded: no capital at risk, no return
            continue
        if timing == "end_of_day":
            r = (d.nav - d.flow) / prev - 1.0
        else:
            base = prev + d.flow
            if base <= 0:
                prev = d.nav
                continue
            r = d.nav / base - 1.0
        prev = d.nav
        if math.isfinite(r):
            out.append(DailyReturn(d.day, r, seg))
    return out


def link(returns: Sequence[float]) -> float:
    g = 1.0
    for r in returns:
        g *= 1.0 + r
    return g - 1.0


def index_series(returns: Sequence[DailyReturn]) -> list[tuple[date, float]]:
    """Chain-linked growth index per segment; each segment restarts at 1.0."""
    out: list[tuple[date, float]] = []
    level: float = 1.0
    seg: int | None = None
    for r in returns:
        if seg is not None and r.segment != seg:
            level = 1.0
        seg = r.segment
        level *= 1.0 + r.value
        out.append((r.day, level))
    return out


def annualised_volatility(returns: Sequence[float]) -> float | None:
    if len(returns) < MIN_RETURNS_FOR_VOL:
        return None
    m = sum(returns) / len(returns)
    var = sum((r - m) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(var) * math.sqrt(TRADING_DAYS)


def max_drawdown(returns: Sequence[float]) -> float | None:
    """Most negative peak-to-trough fall of the linked index (<= 0)."""
    if len(returns) < MIN_RETURNS_FOR_DRAWDOWN:
        return None
    level, peak, worst = 1.0, 1.0, 0.0
    for r in returns:
        level *= 1.0 + r
        peak = max(peak, level)
        worst = min(worst, level / peak - 1.0)
    return worst


def annualise(total: float, start: date, end: date) -> float | None:
    days = (end - start).days
    if days < MIN_DAYS_FOR_ANNUALISATION or total <= -1.0:
        return None
    return float((1.0 + total) ** (365.0 / days) - 1.0)


@dataclass(frozen=True)
class PeriodReturn:
    label: str
    first_day: date
    last_day: date
    value: float
    days: int
    complete: bool


def period_returns(
    returns: Sequence[DailyReturn], weekdays: Sequence[date], kind: Literal["month", "year"]
) -> list[PeriodReturn]:
    """Compound daily returns per calendar period.

    A period is `complete` only if every weekday of that period has a return from one unbroken
    segment, i.e. the account was already invested and valued from the period's first weekday.
    """
    if not returns:
        return []

    def key(d: date) -> str:
        return f"{d.year}" if kind == "year" else f"{d.year}-{d.month:02d}"

    by_key: dict[str, list[DailyReturn]] = {}
    for r in returns:
        by_key.setdefault(key(r.day), []).append(r)
    wd_count: dict[str, int] = {}
    for d in weekdays:
        wd_count[key(d)] = wd_count.get(key(d), 0) + 1
    out: list[PeriodReturn] = []
    for k, rs in by_key.items():
        complete = len({r.segment for r in rs}) == 1 and len(rs) == wd_count.get(k, -1)
        out.append(
            PeriodReturn(k, rs[0].day, rs[-1].day, link([r.value for r in rs]), len(rs), complete)
        )
    return out

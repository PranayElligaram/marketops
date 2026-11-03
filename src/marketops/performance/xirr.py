"""Money-weighted return (XIRR) that refuses to answer when the answer is not unique or defined.

Cash flows are signed from the investor's side: money paid in is negative, money taken out and the
terminal valuation are positive. The rate solves sum(a_i * (1 + r) ** (-t_i)) = 0 with t in years
(actual/365 from the first flow). The equation can have several roots; all are searched for and a
rate is returned only when exactly one exists.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

XirrStatus = Literal["ok", "ambiguous", "no_solution", "insufficient", "invalid"]

_X_MIN, _X_MAX, _GRID = math.log(0.001), math.log(1001.0), 4000  # r from -99.9 % to +100,000 %


@dataclass(frozen=True)
class CashFlow:
    when: datetime
    amount: float


class XirrResult(BaseModel):
    status: XirrStatus
    rate: float | None = None
    roots: list[float] = []
    reason: str = ""
    period_days: float | None = None
    flow_count: int = 0
    warnings: list[str] = []


def _result(status: XirrStatus, reason: str, **kw: object) -> XirrResult:
    return XirrResult(status=status, reason=reason, **kw)  # type: ignore[arg-type]


def xirr(flows: Sequence[CashFlow], *, min_days: float = 30.0) -> XirrResult:
    for f in flows:
        if f.when.tzinfo is None:
            return _result("invalid", "A cash flow has a date without a time zone.")
        if not math.isfinite(f.amount):
            return _result("invalid", "A cash flow amount is not a finite number.")
    items = sorted((f for f in flows if f.amount != 0), key=lambda f: f.when)
    n = len(items)
    if n < 2:
        return _result("insufficient", "At least two non-zero cash flows are needed.", flow_count=n)
    if not any(f.amount < 0 for f in items) or not any(f.amount > 0 for f in items):
        return _result(
            "insufficient",
            "Cash flows need both money paid in and money (or value) received.",
            flow_count=n,
        )
    t0 = items[0].when
    years = [(f.when - t0).total_seconds() / 86400.0 / 365.0 for f in items]
    days = years[-1] * 365.0
    if days < min_days:
        return _result(
            "insufficient",
            f"Period is {days:.0f} days; an annualised rate needs at least {min_days:.0f}.",
            period_days=days,
            flow_count=n,
        )
    scale = max(abs(f.amount) for f in items)
    amounts = [f.amount / scale for f in items]

    def g(x: float) -> float:
        return sum(a * math.exp(-x * t) for a, t in zip(amounts, years, strict=True))

    try:
        xs = [_X_MIN + (_X_MAX - _X_MIN) * i / _GRID for i in range(_GRID + 1)]
        vals = [g(x) for x in xs]
        roots: list[float] = []
        for i in range(_GRID):
            lo, hi, flo, fhi = xs[i], xs[i + 1], vals[i], vals[i + 1]
            if flo == 0.0:
                roots.append(lo)
            elif flo * fhi < 0:
                for _ in range(200):
                    mid = (lo + hi) / 2
                    fm = g(mid)
                    if (fm < 0) == (flo < 0):
                        lo, flo = mid, fm
                    else:
                        hi = mid
                roots.append((lo + hi) / 2)
        if vals[-1] == 0.0:
            roots.append(xs[-1])
    except OverflowError:
        return _result("invalid", "Cash flow dates are too far apart to evaluate.", flow_count=n)
    rates = sorted({round(math.exp(x) - 1.0, 10) for x in roots})
    common = {"period_days": days, "flow_count": n}
    if not rates:
        return _result(
            "no_solution",
            "No rate between -99.9% and +100,000% balances these cash flows (for example, a "
            "terminal value of zero).",
            **common,
        )
    if len(rates) > 1:
        return _result(
            "ambiguous",
            f"{len(rates)} different rates solve these cash flows, so no single return is defined.",
            roots=rates,
            **common,
        )
    warnings = ["Annualised from a period shorter than one year."] if days < 365 else []
    return _result("ok", "", rate=rates[0], roots=rates, warnings=warnings, **common)

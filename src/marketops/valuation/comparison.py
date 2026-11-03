"""Portfolio versus contribution-matched benchmark: like-for-like dates, currency and cash flows.

Pure (no I/O). The benchmark is what the same external deposits and withdrawals would be worth had
every pound been put into a total-return benchmark price series at the close of its execution day
(`benchmark.simulate_benchmark`).

Comparison window: only dates on which the portfolio is valued AND the benchmark is priced, from
the first common date on/after the first external flow to the last common date. Both sides are
read at the same close, in the same currency, with the same flows. Anything that would break that
(gaps, stale or missing benchmark data, contribution mismatch, over-withdrawal) makes the whole
comparison `unavailable` with the exact reason; nothing is filled, interpolated or substituted.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from marketops.performance.xirr import CashFlow, xirr
from marketops.valuation import twr
from marketops.valuation.benchmark import (
    BenchmarkPoint,
    BenchmarkPrice,
    ExternalFlow,
    simulate_benchmark,
)

Status = Literal["available", "partial", "unavailable"]

MAX_EXEC_DELAY_DAYS = 5  # a flow waiting longer than this for a benchmark price means a data gap
MAX_END_LAG_DAYS = 5  # benchmark may end at most this many days before the portfolio
MAX_PRICE_GAP_DAYS = 5  # longest tolerated gap between benchmark prices inside the window
MAX_DAILY_MOVE = 0.25  # a larger one-day benchmark move is treated as corrupt data
CONTRIBUTION_TOLERANCE = Decimal("0.01")
BENCHMARK_NAME = "S&P 500 (SPY total return, contribution-matched)"

CONVENTIONS: dict[str, str] = {
    "benchmark": "SPY dividend-adjusted close (total-return proxy) converted to the account "
    "currency with the ECB reference rate of the same date; dividends are inside the price and "
    "are never added again.",
    "flows": "The portfolio's external deposits and withdrawals are replayed into the benchmark "
    "at the close of the same day, or the next priced day if the benchmark did not trade.",
    "window": "Only days on which the portfolio is valued and the benchmark is priced; both are "
    "read at the same close.",
    "return": "Time-weighted returns chain daily returns with flows removed; both sides use the "
    "same end-of-day flow timing.",
    "xirr": "Closing-price XIRR: the same dated flows for both sides and a terminal value at the "
    "same last common close. It is not the live-snapshot XIRR, which ends at a live balance.",
}
LIMITATIONS: tuple[str, ...] = (
    "SPY is an ETF proxy for the S&P 500, not the index: it includes about 0.09% annual expense "
    "ratio and tracking error, and the dividend adjustment follows the data provider's method.",
    "The benchmark assumes fractional units and no trading costs, spreads or taxes; the real "
    "account has its own fees, fills and tax treatment.",
    "A UK investor in SPY holds US dollar exposure, so the result mixes US equity and USD/GBP "
    "moves.",
    "ECB reference rates are fixed around 16:00 CET while US markets close later; same-date "
    "pairing is used and no intraday timing is modelled.",
    "The portfolio holds a different mix (and risk) than the S&P 500; outperformance or "
    "underperformance is not evidence of skill. Past performance does not predict the future.",
)


class BenchmarkSource(BaseModel):
    symbol: str
    provider: str | None
    basis: str
    quote_currency: str | None
    first_day: str | None
    last_day: str | None
    rows_usable: int
    rows_dropped_invalid_price: int
    rows_dropped_missing_fx: int
    retrieved_utc: str | None
    sha256: str | None
    age_days: int | None = None


class ComparisonPoint(BaseModel):
    date: str
    portfolio_value: float
    benchmark_value: float
    net_invested: float
    portfolio_twr: float
    benchmark_twr: float
    portfolio_drawdown: float
    benchmark_drawdown: float


class CmpMetric(BaseModel):
    name: str
    status: Status
    value: float | None = None
    unit: str | None = None
    basis: str
    blockers: list[str] = []


class BenchmarkComparison(BaseModel):
    status: Status
    name: str = BENCHMARK_NAME
    currency: str | None = None
    reasons: list[str] = []
    window_start: str | None = None
    window_end: str | None = None
    common_days: int = 0
    source: BenchmarkSource | None = None
    series: list[ComparisonPoint] = []
    metrics: list[CmpMetric] = []
    conventions: dict[str, str] = CONVENTIONS
    limitations: list[str] = list(LIMITATIONS)


@dataclass(frozen=True)
class PortfolioDay:
    day: date
    nav: Decimal | None
    flow: Decimal  # external flow dated this day (deposit +, withdrawal -)
    net_invested: Decimal  # cumulative, including this day


@dataclass
class BenchmarkInput:
    prices: list[BenchmarkPrice]
    source: BenchmarkSource | None
    problems: list[str] = field(default_factory=list)


def unavailable(
    why: str, source: BenchmarkSource | None = None, ccy: str | None = None
) -> BenchmarkComparison:
    return BenchmarkComparison(status="unavailable", currency=ccy, reasons=[why], source=source)


def _levels(rows: Sequence[twr.DayValue]) -> dict[date, float]:
    """Chain-linked growth level (1.0 at the first valued day) for every valued day."""
    out: dict[date, float] = {}
    rets = {r.day: r.value for r in twr.daily_returns(rows, "end_of_day")}
    level: float | None = None
    for row in rows:
        if row.nav is None:
            continue
        if level is None:
            level = 1.0
        else:
            level *= 1.0 + rets.get(row.day, 0.0)
        out[row.day] = level
    return out


def _drawdowns(levels: Sequence[float]) -> list[float]:
    peak, out = -math.inf, []
    for v in levels:
        peak = max(peak, v)
        out.append(v / peak - 1.0)
    return out


def _stepped(levels: Sequence[float]) -> list[float]:
    return [b / a - 1.0 for a, b in zip(levels, levels[1:], strict=False)]


def _at_noon(d: date) -> datetime:
    return datetime.combine(d, time(12), tzinfo=UTC)


def compare(
    days: Sequence[PortfolioDay],
    bench: BenchmarkInput | None,
    ccy: str | None,
    reconciled: bool,
    portfolio_blockers: Sequence[str] = (),
) -> BenchmarkComparison:
    if bench is None or not bench.prices:
        why = " ".join(bench.problems) if bench and bench.problems else "No benchmark data."
        return unavailable(
            why + " Download benchmark data to enable the comparison.",
            bench.source if bench else None,
            ccy,
        )
    src = bench.source
    if bench.problems:
        return unavailable(" ".join(bench.problems), src, ccy)
    valued = [d for d in days if d.nav is not None]
    flows = [d for d in days if d.flow != 0]
    if not valued or not flows:
        return unavailable("The portfolio has no valued days or no external flows.", src, ccy)

    prices = bench.prices
    pdays = [p.day for p in prices]
    first_flow = flows[0].day
    last_valued = valued[-1].day
    if pdays[-1] < last_valued and (last_valued - pdays[-1]).days > MAX_END_LAG_DAYS:
        return unavailable(
            f"Benchmark prices end on {pdays[-1]} but the portfolio is valued to {last_valued}; "
            "refresh the benchmark data.",
            src,
            ccy,
        )
    # Every flow must execute within a few days of its own date, or there is a data gap.
    end = min(last_valued, pdays[-1])
    used_days = [d for d in days if d.day <= end]
    used_flows = [d for d in used_days if d.flow != 0]
    for f in used_flows:
        i = bisect_left(pdays, f.day)
        if i == len(pdays) or (pdays[i] - f.day).days > MAX_EXEC_DELAY_DAYS:
            nxt = pdays[i] if i < len(pdays) else "never"
            return unavailable(
                f"No benchmark price within {MAX_EXEC_DELAY_DAYS} days after the flow on "
                f"{f.day} (next priced day: {nxt}).",
                src,
                ccy,
            )
    if not used_flows:
        return unavailable("No external flow falls inside the benchmark price range.", src, ccy)

    priced = set(pdays)
    window_days = [d for d in used_days if d.nav is not None and d.day >= first_flow]
    unvalued_inside = (
        [d.day for d in used_days if d.nav is None and d.day >= window_days[0].day]
        if window_days
        else []
    )
    if unvalued_inside:
        return unavailable(
            f"{len(unvalued_inside)} weekday(s) between {window_days[0].day} and {end} could not "
            f"be valued (first: {unvalued_inside[0]}); the portfolio index would be broken.",
            src,
            ccy,
        )
    common = [d for d in window_days if d.day in priced]
    if len(common) < 2:
        return unavailable(
            "Fewer than two dates have both a portfolio value and a benchmark price.", src, ccy
        )
    start_day, end_day = common[0].day, common[-1].day
    in_window = [p for p in prices if start_day <= p.day <= end_day]
    for a, b in zip(in_window, in_window[1:], strict=False):
        if (b.day - a.day).days > MAX_PRICE_GAP_DAYS:
            return unavailable(f"Benchmark prices have a gap from {a.day} to {b.day}.", src, ccy)
        move = float(b.price / a.price) - 1.0
        if abs(move) > MAX_DAILY_MOVE:
            return unavailable(
                f"Benchmark moves {move:+.0%} between {a.day} and {b.day}: corrupt data.",
                src,
                ccy,
            )

    try:
        sim = simulate_benchmark(
            [p for p in prices if p.day <= end_day],
            [ExternalFlow(f.day, f.flow) for f in used_flows if f.day <= end_day],
        )
    except ValueError as exc:
        return unavailable(f"The benchmark cannot replay the portfolio's flows: {exc}", src, ccy)
    sim_by_day = {p.day: p for p in sim}

    # contribution check: the benchmark must have received exactly what the portfolio received
    for d in common:
        diff = abs(sim_by_day[d.day].net_contributions - d.net_invested)
        if diff > CONTRIBUTION_TOLERANCE:
            return unavailable(
                f"Net contributions differ on {d.day} (portfolio {d.net_invested}, benchmark "
                f"{sim_by_day[d.day].net_contributions}); the comparison is not like-for-like.",
                src,
                ccy,
            )

    # time-weighted levels
    carried: list[twr.DayValue] = []
    pending = 0.0
    for d in used_days:
        pending += float(d.flow)
        if d.nav is None:
            carried.append(twr.DayValue(d.day, None, 0.0))
        else:
            carried.append(twr.DayValue(d.day, float(d.nav), pending))
            pending = 0.0
    p_levels = _levels([c for c in carried if c.day >= first_flow or c.nav is None])
    sim_rows = [
        twr.DayValue(p.day, float(p.value), float(p.net_contributions - prev))
        for p, prev in zip(sim, [Decimal(0), *[q.net_contributions for q in sim]], strict=False)
    ]
    b_levels = _levels(sim_rows)
    if any(d.day not in p_levels or d.day not in b_levels for d in common):
        return unavailable("A common date has no time-weighted level.", src, ccy)

    p0, b0 = p_levels[start_day], b_levels[start_day]
    p_lv = [p_levels[d.day] / p0 for d in common]
    b_lv = [b_levels[d.day] / b0 for d in common]
    p_dd, b_dd = _drawdowns(p_lv), _drawdowns(b_lv)
    series = [
        ComparisonPoint(
            date=d.day.isoformat(),
            portfolio_value=float(d.nav or 0),
            benchmark_value=float(sim_by_day[d.day].value),
            net_invested=float(d.net_invested),
            portfolio_twr=pl - 1.0,
            benchmark_twr=bl - 1.0,
            portfolio_drawdown=pd_,
            benchmark_drawdown=bd_,
        )
        for d, pl, bl, pd_, bd_ in zip(common, p_lv, b_lv, p_dd, b_dd, strict=True)
    ]

    blockers = list(portfolio_blockers)
    if not reconciled:
        blockers.append("The portfolio valuation is not reconciled to the live account.")
    ok: Status = "available" if not blockers else "partial"
    metrics = _metrics(
        ok, blockers, common, sim_by_day, p_lv, b_lv, start_day, end_day, used_flows, ccy
    )
    return BenchmarkComparison(
        status=ok,
        currency=ccy,
        reasons=blockers,
        window_start=start_day.isoformat(),
        window_end=end_day.isoformat(),
        common_days=len(common),
        source=src,
        series=series,
        metrics=metrics,
    )


def _metrics(
    ok: Status,
    blockers: list[str],
    common: Sequence[PortfolioDay],
    sim: dict[date, BenchmarkPoint],
    p_lv: Sequence[float],
    b_lv: Sequence[float],
    start: date,
    end: date,
    flows: Sequence[PortfolioDay],
    ccy: str | None,
) -> list[CmpMetric]:
    out: list[CmpMetric] = []

    def add(name: str, value: float | None, unit: str, basis: str, why: str = "") -> None:
        if value is None:
            out.append(
                CmpMetric(name=name, status="unavailable", unit=unit, basis=basis, blockers=[why])
            )
        else:
            out.append(
                CmpMetric(
                    name=name, status=ok, value=value, unit=unit, basis=basis, blockers=blockers
                )
            )

    last = common[-1]
    bench_value = float(sim[last.day].value)
    nav = float(last.nav or 0)
    add("portfolio_value", nav, ccy or "", "Portfolio value at the last common close.")
    add(
        "benchmark_value",
        bench_value,
        ccy or "",
        "What the same flows would be worth in the benchmark.",
    )
    add(
        "net_invested_capital",
        float(last.net_invested),
        ccy or "",
        "Deposits minus withdrawals, identical for both.",
    )
    add("value_difference", nav - bench_value, ccy or "", "Portfolio value minus benchmark value.")
    p_tot, b_tot = p_lv[-1] - 1.0, b_lv[-1] - 1.0
    add("portfolio_twr", p_tot, "fraction", "Time-weighted return over the comparison window.")
    add("benchmark_twr", b_tot, "fraction", "Time-weighted return over the comparison window.")
    add(
        "twr_difference", p_tot - b_tot, "fraction", "Portfolio minus benchmark, percentage points."
    )
    p_ann, b_ann = twr.annualise(p_tot, start, end), twr.annualise(b_tot, start, end)
    short = "Period shorter than one year."
    add("portfolio_annualised_twr", p_ann, "fraction/yr", "Geometric annualisation.", short)
    add("benchmark_annualised_twr", b_ann, "fraction/yr", "Geometric annualisation.", short)
    add(
        "annualised_difference",
        None if p_ann is None or b_ann is None else p_ann - b_ann,
        "fraction/yr",
        "Portfolio minus benchmark, annualised.",
        short,
    )
    p_mdd, b_mdd = twr.max_drawdown(_stepped(p_lv)), twr.max_drawdown(_stepped(b_lv))
    few = "Fewer than 20 common-day returns."
    add("portfolio_max_drawdown", p_mdd, "fraction", "Peak-to-trough fall on common days.", few)
    add("benchmark_max_drawdown", b_mdd, "fraction", "Peak-to-trough fall on common days.", few)
    add(
        "drawdown_difference",
        None if p_mdd is None or b_mdd is None else p_mdd - b_mdd,
        "fraction",
        "Portfolio minus benchmark maximum drawdown (negative = deeper portfolio fall).",
        few,
    )
    # closing-price XIRR: same dated flows, same terminal date for both sides
    cfs = [CashFlow(_at_noon(f.day), -float(f.flow)) for f in flows if f.day <= end]
    p_x = xirr([*cfs, CashFlow(_at_noon(end), nav)])
    b_x = xirr([*cfs, CashFlow(_at_noon(end), bench_value)])
    for name, res in (("portfolio_xirr_closing", p_x), ("benchmark_xirr_closing", b_x)):
        add(
            name,
            res.rate if res.status == "ok" else None,
            "fraction/yr",
            "Same dated flows, terminal value at the last common close.",
            res.reason,
        )
    if p_x.status == "ok" and b_x.status == "ok":
        assert p_x.rate is not None and b_x.rate is not None
        add("xirr_difference", p_x.rate - b_x.rate, "fraction/yr", "Portfolio minus benchmark.")
    else:
        add(
            "xirr_difference",
            None,
            "fraction/yr",
            "Portfolio minus benchmark.",
            "An XIRR is unavailable.",
        )
    return out

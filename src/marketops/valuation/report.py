"""Verified historical valuation report: NAV, net invested capital, time-weighted performance.

Pure assembly over stored history (PostgreSQL rows), cached prices and FX. It performs no network
access. Each metric is `available` only when its inputs reconcile; otherwise `partial` with the
blockers or `unavailable` with the reason. Nothing is interpolated or invented.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from marketops.performance.holdings import QTY_TOLERANCE
from marketops.valuation import twr
from marketops.valuation.cache import load_series
from marketops.valuation.comparison import (
    BenchmarkComparison,
    BenchmarkInput,
    PortfolioDay,
    compare,
)
from marketops.valuation.fx import FxTable
from marketops.valuation.nav import DayNav, PriceSeries, build_events, compute_daily_nav
from marketops.valuation.symbols import (
    MappingVerdict,
    candidate_symbol,
    coverage_range,
    verify_mapping,
)

NAV_MATCH = 0.015  # last close vs the live snapshot differ by price movement, not error
NAV_DRIFT = 0.04
TIMING_SENSITIVE = 0.005  # TWR difference between flow-timing conventions worth flagging

Status = Literal["available", "partial", "unavailable"]

CONVENTIONS: dict[str, str] = {
    "event_date": "UTC calendar date of each record; weekend events roll to the next weekday.",
    "valuation_days": "Every Monday-Friday from the first record; end-of-day NAV.",
    "price": "Provider split-adjusted close restated to the as-traded price; never adj_close.",
    "price_staleness": "Last close up to 5 calendar days old, never across a split.",
    "fx": "ECB euro reference rate on or before the day (up to 5 days old), triangulated via EUR.",
    "minor_units": "GBX/GBp prices are divided by 100.",
    "twr_flows": "External flows occur after the close of their day (sensitivity vs start of day "
    "is reported).",
    "dividends": "Counted only as cash received; prices are not dividend-adjusted.",
    "benchmark_exception": "Only the S&P 500 benchmark (SPY) uses dividend-adjusted prices, "
    "as a total-return proxy.",
}


class MappingRow(BaseModel):
    ticker: str
    symbol: str | None
    source: str
    status: str
    fills_checked: int
    fills_passed: int
    provider_currency: str | None
    instrument_currency: str | None
    price_first: str | None
    price_last: str | None
    reasons: list[str]


class SeriesPoint(BaseModel):
    date: str
    nav: float | None
    cash: float | None
    holdings_value: float | None
    net_invested: float
    twr_index: float | None = None


class PeriodRow(BaseModel):
    period: str
    twr: float
    days: int
    complete: bool


class ValMetric(BaseModel):
    name: str
    status: Status
    value: float | None = None
    unit: str | None = None
    basis: str
    blockers: list[str] = []


class NavReconciliation(BaseModel):
    status: Literal["match", "drift", "mismatch", "not_checked"]
    valuation_date: str | None
    difference_pct: float | None
    holdings: list[dict[str, Any]]
    quantity_status: Literal["reconciled", "mismatch", "not_checked"]
    reasons: list[str]


class Coverage(BaseModel):
    first_day: str | None
    last_day: str | None
    weekdays: int
    valued_days: int
    unvalued_days: int
    stale_price_days: int
    unvalued_reasons: dict[str, int]
    fx_currencies: dict[str, str]


class HistoricalValuation(BaseModel):
    as_of_utc: str
    account_currency: str | None
    overall_status: Literal["verified", "partial", "unavailable"]
    conventions: dict[str, str]
    mapping: list[MappingRow]
    coverage: Coverage
    series: list[SeriesPoint]
    monthly: list[PeriodRow]
    annual: list[PeriodRow]
    metrics: list[ValMetric]
    reconciliation: NavReconciliation
    notes: list[str]
    benchmark: BenchmarkComparison | None = None


def _f(v: Decimal | None) -> float | None:
    return None if v is None else float(v)


def _reason_code(r: str) -> str:
    return r.split(":", 1)[0]


def _blockers(history_verified: bool, extra: Sequence[str]) -> list[str]:
    out = list(extra)
    if not history_verified:
        out.append("Cash ledger or holdings reconstruction is not reconciled to the broker.")
    return out


def build_valuation(
    *,
    fills: Sequence[Mapping[str, Any]],
    tx: Sequence[Mapping[str, Any]],
    divs: Sequence[Mapping[str, Any]],
    ccy: str | None,
    as_of: Any,
    price_root: Path | None,
    fx: FxTable | None,
    overrides: Mapping[str, str] | None,
    history_verified: bool,
    ledger_cash: float | None,
    broker_positions: Mapping[str, float] | None,
    broker_values: Mapping[str, float] | None,
    broker_total: float | None,
    series_override: Mapping[str, PriceSeries] | None = None,
    benchmark: BenchmarkInput | None = None,
) -> HistoricalValuation:
    notes: list[str] = []
    empty_cov = Coverage(
        first_day=None,
        last_day=None,
        weekdays=0,
        valued_days=0,
        unvalued_days=0,
        stale_price_days=0,
        unvalued_reasons={},
        fx_currencies={},
    )
    no_rec = NavReconciliation(
        status="not_checked",
        valuation_date=None,
        difference_pct=None,
        holdings=[],
        quantity_status="not_checked",
        reasons=[],
    )

    def unavailable(why: str) -> HistoricalValuation:
        return HistoricalValuation(
            as_of_utc=as_of.isoformat(),
            account_currency=ccy,
            overall_status="unavailable",
            conventions=CONVENTIONS,
            mapping=[],
            coverage=empty_cov,
            series=[],
            monthly=[],
            annual=[],
            metrics=[],
            reconciliation=no_rec,
            notes=[why],
        )

    if not ccy:
        return unavailable("Account currency is unknown.")
    events, corp = build_events(tx, divs, fills, ccy)
    if not events:
        return unavailable("No usable history records.")

    # --- instruments and mapping verification (fully sold instruments included)
    tickers = sorted({f["ticker"] for f in fills if f["category"] in ("buy", "sell")})
    prices: dict[str, PriceSeries] = dict(series_override or {})
    verdicts: dict[str, MappingVerdict] = {}
    rows: list[MappingRow] = []
    for t in tickers:
        cand = candidate_symbol(t, overrides)
        s = prices.get(cand.symbol or "") if cand.symbol else None
        if s is None and cand.symbol and price_root is not None:
            s = load_series(price_root, cand.symbol)
            if s is not None:
                prices[cand.symbol] = s
        v = verify_mapping(
            t, cand, None if s is None else s.frame, None if s is None else s.currency, fills
        )
        verdicts[t] = v
        rng = coverage_range(None if s is None else s.frame)
        rows.append(
            MappingRow(
                ticker=t,
                symbol=v.symbol,
                source=v.source,
                status=v.status,
                fills_checked=v.fills_checked,
                fills_passed=v.fills_passed,
                provider_currency=v.provider_currency,
                instrument_currency=v.instrument_currency,
                price_first=rng[0].isoformat() if rng else None,
                price_last=rng[1].isoformat() if rng else None,
                reasons=v.reasons,
            )
        )

    # --- valuation window: through the last day that has any price coverage
    as_of_day: date = as_of.date()
    days, final_qty = compute_daily_nav(events, corp, verdicts, prices, fx, ccy, as_of_day)
    return _assemble(
        days,
        final_qty,
        rows,
        ccy,
        as_of,
        history_verified,
        ledger_cash,
        broker_positions,
        broker_values,
        broker_total,
        fx,
        notes,
        benchmark,
    )


def _assemble(
    days: list[DayNav],
    final_qty: Mapping[str, Decimal],
    mapping: list[MappingRow],
    ccy: str,
    as_of: Any,
    history_verified: bool,
    ledger_cash: float | None,
    broker_positions: Mapping[str, float] | None,
    broker_values: Mapping[str, float] | None,
    broker_total: float | None,
    fx: FxTable | None,
    notes: list[str],
    benchmark: BenchmarkInput | None = None,
) -> HistoricalValuation:
    valued = [d for d in days if d.nav is not None]
    reasons = Counter(_reason_code(r) for d in days if d.nav is None for r in d.reasons)
    fx_status: dict[str, str] = {}
    for d in valued:
        for p in d.positions:
            if p.local_currency != ccy:
                fx_status[p.local_currency] = "ecb"
    for m in mapping:
        c = (m.provider_currency or "").upper()
        if c and c not in fx_status and c not in ("GBP", "GBX", ccy):
            fx_status[c] = "ecb_available" if fx and fx.coverage(c) else "missing"

    coverage = Coverage(
        first_day=days[0].day.isoformat() if days else None,
        last_day=days[-1].day.isoformat() if days else None,
        weekdays=len(days),
        valued_days=len(valued),
        unvalued_days=len(days) - len(valued),
        stale_price_days=sum(1 for d in valued if d.stale_price_positions),
        unvalued_reasons=dict(reasons),
        fx_currencies=fx_status,
    )

    # --- time-weighted performance
    # flows on unvalued days must not vanish: carry them into the next valued day
    carried: list[twr.DayValue] = []
    pending = 0.0
    for d in days:
        pending += float(d.flow)
        if d.nav is None:
            carried.append(twr.DayValue(d.day, None, 0.0))
        else:
            carried.append(twr.DayValue(d.day, float(d.nav), pending))
            pending = 0.0
    rets_end = twr.daily_returns(carried, "end_of_day")
    rets_start = twr.daily_returns(carried, "start_of_day")
    index = dict(twr.index_series(rets_end))
    series = [
        SeriesPoint(
            date=d.day.isoformat(),
            nav=_f(d.nav),
            cash=_f(d.cash),
            holdings_value=_f(d.holdings_value),
            net_invested=float(d.net_invested),
            twr_index=index.get(d.day),
        )
        for d in days
    ]
    weekdays = [d.day for d in days]
    monthly = [
        PeriodRow(period=p.label, twr=p.value, days=p.days, complete=p.complete)
        for p in twr.period_returns(rets_end, weekdays, "month")
    ]
    annual = [
        PeriodRow(period=p.label, twr=p.value, days=p.days, complete=p.complete)
        for p in twr.period_returns(rets_end, weekdays, "year")
    ]

    segments = sorted({r.segment for r in rets_end})
    first_valued = days.index(valued[0]) if valued else 0
    single_segment = len(segments) == 1 and all(d.nav is not None for d in days[first_valued:])
    longest = None
    if segments:
        longest = max(segments, key=lambda s: sum(1 for r in rets_end if r.segment == s))
    seg_rets = [r for r in rets_end if r.segment == longest]
    values = [r.value for r in seg_rets]
    total = twr.link(values) if values else None
    total_start = (
        twr.link([r.value for r in rets_start if r.segment == longest]) if values else None
    )

    metrics: list[ValMetric] = []
    blockers = list(
        dict.fromkeys(
            _blockers(
                history_verified,
                [f"{n} weekday(s) could not be valued: {k}." for k, n in reasons.items()],
            )
        )
    )
    last = valued[-1] if valued else None
    metrics.append(
        ValMetric(
            name="portfolio_nav",
            status="available" if last and not blockers else ("partial" if last else "unavailable"),
            value=_f(last.nav) if last else None,
            unit=ccy,
            basis="Holdings at the last valid close plus reconstructed cash.",
            blockers=blockers,
        )
    )
    metrics.append(
        ValMetric(
            name="net_invested_capital",
            status="available"
            if days and history_verified
            else "partial"
            if days
            else "unavailable",
            value=float(days[-1].net_invested) if days else None,
            unit=ccy,
            basis="Deposits minus withdrawals to the last valuation day.",
            blockers=[] if history_verified else _blockers(False, []),
        )
    )
    if last:
        gain = last.nav - last.net_invested if last.nav is not None else None
        metrics.append(
            ValMetric(
                name="net_gain_on_invested_capital",
                status="available" if not blockers else "partial",
                value=_f(gain),
                unit=ccy,
                basis="NAV minus net invested capital (dividends, interest, fees, price moves).",
                blockers=blockers,
            )
        )
    twr_ok = bool(values) and single_segment and history_verified
    twr_blockers = list(blockers)
    if len(segments) > 1:
        twr_blockers.append(
            f"{len(segments)} unbroken segments; the figure covers the longest only."
        )
    if not values:
        twr_blockers.append("Fewer than two consecutive valued days after the account was funded.")
    metrics.append(
        ValMetric(
            name="time_weighted_return",
            status="available" if twr_ok else "partial" if values else "unavailable",
            value=total,
            unit="fraction",
            basis="Chain-linked daily returns, flows after the close of their day.",
            blockers=[] if twr_ok else twr_blockers,
        )
    )
    if values and total is not None and total_start is not None:
        diff = abs(total - total_start)
        metrics.append(
            ValMetric(
                name="twr_flow_timing_sensitivity",
                status="available" if diff <= TIMING_SENSITIVE else "partial",
                value=total_start - total,
                unit="fraction",
                basis="TWR with flows at the start of the day minus TWR with flows at the end.",
                blockers=[]
                if diff <= TIMING_SENSITIVE
                else ["Result depends on intraday flow timing."],
            )
        )
        first_r, last_r = seg_rets[0].day, seg_rets[-1].day
        ann = twr.annualise(total, first_r, last_r)
        metrics.append(
            ValMetric(
                name="annualised_twr",
                status="available" if ann is not None and twr_ok else "unavailable",
                value=ann,
                unit="fraction/yr",
                basis="Geometric annualisation, only for periods of at least one year.",
                blockers=[] if ann is not None else ["Period shorter than one year."],
            )
        )
        vol = twr.annualised_volatility(values)
        metrics.append(
            ValMetric(
                name="annualised_volatility",
                status="available"
                if vol is not None and twr_ok
                else "partial"
                if vol is not None
                else "unavailable",
                value=vol,
                unit="fraction/yr",
                basis="Sample standard deviation of daily TWR returns x sqrt(252).",
                blockers=[] if vol is not None else ["Fewer than 60 daily returns."],
            )
        )
        dd = twr.max_drawdown(values)
        metrics.append(
            ValMetric(
                name="max_drawdown",
                status="available"
                if dd is not None and twr_ok
                else "partial"
                if dd is not None
                else "unavailable",
                value=dd,
                unit="fraction",
                basis="Largest peak-to-trough fall of the TWR index (flows removed).",
                blockers=[] if dd is not None else ["Fewer than 20 daily returns."],
            )
        )

    recon = _reconcile(
        last, final_qty, ledger_cash, broker_positions, broker_values, broker_total, as_of
    )
    overall: Literal["verified", "partial", "unavailable"]
    if not valued:
        overall = "unavailable"
    elif (
        twr_ok
        and recon.status in ("match", "drift")
        and recon.quantity_status == "reconciled"
        and not blockers
    ):
        overall = "verified"
    else:
        overall = "partial"
    notes += [
        "XIRR (money-weighted) is reported separately by /performance/historical; the "
        "time-weighted figures here ignore the timing and size of deposits.",
        "Price data is for personal research use; FX is ECB reference data.",
    ]
    return HistoricalValuation(
        as_of_utc=as_of.isoformat(),
        account_currency=ccy,
        overall_status=overall,
        conventions=CONVENTIONS,
        mapping=mapping,
        coverage=coverage,
        series=series,
        monthly=sorted(monthly, key=lambda p: p.period),
        annual=sorted(annual, key=lambda p: p.period),
        metrics=metrics,
        reconciliation=recon,
        notes=notes,
        benchmark=compare(
            [PortfolioDay(d.day, d.nav, d.flow, d.net_invested) for d in days],
            benchmark,
            ccy,
            overall == "verified",
            blockers,
        )
        if benchmark is not None
        else None,
    )


def _reconcile(
    last: DayNav | None,
    final_qty: Mapping[str, Decimal],
    ledger_cash: float | None,
    broker_positions: Mapping[str, float] | None,
    broker_values: Mapping[str, float] | None,
    broker_total: float | None,
    as_of: Any,
) -> NavReconciliation:
    if broker_positions is None or last is None or last.nav is None:
        return NavReconciliation(
            status="not_checked",
            valuation_date=None if last is None else last.day.isoformat(),
            difference_pct=None,
            holdings=[],
            quantity_status="not_checked",
            reasons=["No live snapshot or no valued day to compare."],
        )
    reasons: list[str] = []
    qty_ok = True
    for t in sorted(set(final_qty) | set(broker_positions)):
        a = final_qty.get(t, Decimal(0))
        b = Decimal(str(broker_positions.get(t, 0.0)))
        if abs(a - b) > QTY_TOLERANCE:
            qty_ok = False
            reasons.append(f"Split-adjusted quantity differs from the broker for {t}.")
    rows: list[dict[str, Any]] = []
    for p in last.positions:
        bv = (broker_values or {}).get(p.ticker)
        rows.append(
            {
                "ticker": p.ticker,
                "valuation_date": p.price_date.isoformat(),
                "difference_pct": None
                if not bv
                else float((p.value - Decimal(str(bv))) / Decimal(str(bv))),
            }
        )
    status: Literal["match", "drift", "mismatch", "not_checked"] = "not_checked"
    diff: float | None = None
    age = (as_of.date() - last.day).days
    if broker_total is None or broker_total == 0:
        reasons.append("Broker total value unavailable.")
    elif age > 4:
        reasons.append("Last valued day is too far from the snapshot for a price comparison.")
    else:
        diff = float((last.nav - Decimal(str(broker_total))) / Decimal(str(broker_total)))
        status = (
            "match" if abs(diff) <= NAV_MATCH else "drift" if abs(diff) <= NAV_DRIFT else "mismatch"
        )
        if status != "match":
            reasons.append(
                "Last-close NAV differs from the live total by more than "
                "ordinary price movement would explain."
                if status == "mismatch"
                else "Last-close NAV differs from the live total (price timing)."
            )
    if ledger_cash is not None and abs(float(last.cash) - ledger_cash) > 1.0:
        reasons.append("Valuation-day cash differs from the reconciled ledger cash.")
    return NavReconciliation(
        status=status,
        valuation_date=last.day.isoformat(),
        difference_pct=diff,
        holdings=rows,
        quantity_status="reconciled" if qty_ok else "mismatch",
        reasons=reasons,
    )

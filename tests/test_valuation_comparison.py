"""Benchmark comparison and benchmark data: independent plain-float reference calculations on
synthetic series (no I/O except tmp_path, no network)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal as D
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from marketops.data.provider import ProviderError
from marketops.valuation.benchmark import BenchmarkPrice
from marketops.valuation.benchmark_data import (
    fetch_benchmark,
    load_benchmark,
    to_account_currency,
)
from marketops.valuation.comparison import (
    BenchmarkComparison,
    BenchmarkInput,
    PortfolioDay,
    compare,
)
from tests.valuation_fixtures import fx_table

START = date(2024, 1, 1)  # Monday


def weekdays(n: int, start: date = START) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def bench_input(
    days: list[date], growth: float = 0.01, start_price: float = 100.0
) -> BenchmarkInput:
    prices = [
        BenchmarkPrice(d, D(str(round(start_price * (1 + growth) ** i, 10))))
        for i, d in enumerate(days)
    ]
    return BenchmarkInput(prices, None)


def portfolio(
    days: list[date], flows: dict[date, float], growth: float = 0.02
) -> list[PortfolioDay]:
    """nav_t = nav_{t-1} * (1 + growth) + flow_t (a flow earns nothing on its own day)."""
    out: list[PortfolioDay] = []
    nav, invested = 0.0, 0.0
    for i, d in enumerate(days):
        f = flows.get(d, 0.0)
        nav = nav * (1 + growth) + f if i else f
        invested += f
        out.append(PortfolioDay(d, D(str(round(nav, 10))), D(str(f)), D(str(invested))))
    return out


def get(c: BenchmarkComparison, name: str) -> Any:
    return next(m for m in c.metrics if m.name == name)


DAYS = weekdays(90)  # 2024-01-01 .. 2024-05-03 (about 124 calendar days)


def test_single_deposit_matches_closed_forms() -> None:
    n = len(DAYS) - 1
    c = compare(portfolio(DAYS, {DAYS[0]: 1000.0}), bench_input(DAYS), "GBP", True)
    assert c.status == "available" and c.window_start == "2024-01-01"
    assert get(c, "portfolio_twr").value == pytest.approx(1.02**n - 1)
    assert get(c, "benchmark_twr").value == pytest.approx(1.01**n - 1)
    assert get(c, "twr_difference").value == pytest.approx(1.02**n - 1.01**n)
    assert get(c, "benchmark_value").value == pytest.approx(1000 * 1.01**n)
    assert get(c, "portfolio_value").value == pytest.approx(1000 * 1.02**n)
    assert get(c, "value_difference").value == pytest.approx(1000 * (1.02**n - 1.01**n))
    days = (DAYS[-1] - DAYS[0]).days
    for name, base in (("portfolio_xirr_closing", 1.02), ("benchmark_xirr_closing", 1.01)):
        assert get(c, name).value == pytest.approx((base**n) ** (365 / days) - 1, rel=1e-6)
    assert get(c, "portfolio_max_drawdown").value == pytest.approx(0)
    assert get(c, "portfolio_annualised_twr").status == "unavailable"  # under a year
    assert all(p.net_invested == 1000 for p in c.series)


def test_mid_deposit_and_withdrawal_match_independent_replay() -> None:
    flows = {DAYS[0]: 1000.0, DAYS[30]: 500.0, DAYS[60]: -300.0}
    c = compare(portfolio(DAYS, flows), bench_input(DAYS), "GBP", True)
    value = 0.0
    for i, d in enumerate(DAYS):
        value = value * 1.01 + flows.get(d, 0.0) if i else flows[d]
        assert c.series[i].benchmark_value == pytest.approx(value)
        assert c.series[i].net_invested == pytest.approx(sum(v for k, v in flows.items() if k <= d))
    # flows never count as return: both TWRs are pure growth
    assert get(c, "benchmark_twr").value == pytest.approx(1.01 ** (len(DAYS) - 1) - 1)
    assert get(c, "portfolio_twr").value == pytest.approx(1.02 ** (len(DAYS) - 1) - 1)
    assert get(c, "net_invested_capital").value == 1200


def test_us_holiday_flow_executes_next_priced_day_and_day_is_not_compared() -> None:
    hol = DAYS[20]
    bench_days = [d for d in DAYS if d != hol]
    flows = {DAYS[0]: 1000.0, hol: 400.0}
    c = compare(portfolio(DAYS, flows), bench_input(bench_days), "GBP", True)
    assert c.status == "available"
    dates = [p.date for p in c.series]
    assert hol.isoformat() not in dates and c.common_days == len(DAYS) - 1
    after, before = (
        c.series[dates.index(DAYS[21].isoformat())],
        c.series[dates.index(DAYS[19].isoformat())],
    )
    assert after.net_invested == 1400 and before.net_invested == 1000
    # the holiday money buys at the next priced day's price, never the previous one
    assert after.benchmark_value == pytest.approx(
        before.benchmark_value * 1.01 + 400
    )  # one priced step: 19 -> 21
    # portfolio TWR across the skipped day still chains both daily portfolio returns
    assert after.portfolio_twr == pytest.approx(1.02**21 - 1)


def test_drawdown_matches_hand_calculation() -> None:
    days = weekdays(30)
    prices = [100.0] * 5 + [110.0] * 5 + [88.0] * 10 + [99.0] * 10  # 110 -> 88 is -20%
    bi = BenchmarkInput(
        [BenchmarkPrice(d, D(str(p))) for d, p in zip(days, prices, strict=True)], None
    )
    c = compare(portfolio(days, {days[0]: 1000.0}, growth=0.0), bi, "GBP", True)
    assert get(c, "benchmark_max_drawdown").value == pytest.approx(88 / 110 - 1)
    assert get(c, "portfolio_max_drawdown").value == pytest.approx(0)
    assert get(c, "drawdown_difference").value == pytest.approx(0 - (88 / 110 - 1))
    assert min(p.benchmark_drawdown for p in c.series) == pytest.approx(-0.2)


def test_annualised_over_more_than_a_year() -> None:
    days = weekdays(300)
    c = compare(portfolio(days, {days[0]: 1000.0}, 0.001), bench_input(days, 0.0005), "GBP", True)
    n, span = len(days) - 1, (days[-1] - days[0]).days
    assert get(c, "portfolio_annualised_twr").value == pytest.approx(1.001 ** (n * 365 / span) - 1)
    assert get(c, "benchmark_annualised_twr").value == pytest.approx(1.0005 ** (n * 365 / span) - 1)
    assert get(c, "annualised_difference").status == "available"


def test_not_reconciled_portfolio_is_partial_not_available() -> None:
    c = compare(portfolio(DAYS, {DAYS[0]: 1000.0}), bench_input(DAYS), "GBP", False)
    assert c.status == "partial" and "not reconciled" in c.reasons[0]
    assert get(c, "benchmark_twr").status == "partial"


# --- refusals: exact reasons, never substituted data ---------------------------------------------


def test_over_withdrawal_makes_comparison_unavailable() -> None:
    days = weekdays(40)
    flows = {days[0]: 100.0, days[10]: -500.0}
    c = compare(portfolio(days, flows, 0.05), bench_input(days, 0.001), "GBP", True)
    assert c.status == "unavailable" and "cannot replay" in c.reasons[0] and not c.series


def test_stale_benchmark_is_unavailable() -> None:
    c = compare(portfolio(DAYS, {DAYS[0]: 1.0}), bench_input(DAYS[:-10]), "GBP", True)
    assert c.status == "unavailable" and "end on" in c.reasons[0]


def test_benchmark_starting_after_first_flow_is_unavailable() -> None:
    c = compare(portfolio(DAYS, {DAYS[0]: 1.0}), bench_input(DAYS[10:]), "GBP", True)
    assert c.status == "unavailable" and "No benchmark price within" in c.reasons[0]


def test_unvalued_portfolio_day_is_unavailable() -> None:
    days = portfolio(DAYS, {DAYS[0]: 1000.0})
    days[40] = PortfolioDay(days[40].day, None, D(0), days[40].net_invested)
    c = compare(days, bench_input(DAYS), "GBP", True)
    assert c.status == "unavailable" and "could not be valued" in c.reasons[0]


def test_benchmark_gap_and_corrupt_move_are_unavailable() -> None:
    gap = [d for i, d in enumerate(DAYS) if not 30 <= i < 40]
    c = compare(portfolio(DAYS, {DAYS[0]: 1.0}), bench_input(gap), "GBP", True)
    assert c.status == "unavailable" and "gap" in c.reasons[0]
    bi = bench_input(DAYS)
    bi.prices[50] = BenchmarkPrice(bi.prices[50].day, bi.prices[50].price * 2)
    c = compare(portfolio(DAYS, {DAYS[0]: 1.0}), bi, "GBP", True)
    assert c.status == "unavailable" and "corrupt" in c.reasons[0]


def test_missing_benchmark_or_problems_are_unavailable_with_reason() -> None:
    p = portfolio(DAYS, {DAYS[0]: 1.0})
    assert "No benchmark data" in compare(p, None, "GBP", True).reasons[0]
    bad = BenchmarkInput([], None, ["No cached SPY benchmark data."])
    assert "No cached SPY" in compare(p, bad, "GBP", True).reasons[0]
    assert compare([], bench_input(DAYS), "GBP", True).status == "unavailable"


def test_no_look_ahead_later_benchmark_prices_do_not_change_earlier_points() -> None:
    flows = {DAYS[0]: 1000.0, DAYS[30]: 200.0}
    full = compare(portfolio(DAYS, flows), bench_input(DAYS), "GBP", True)
    bi = bench_input(DAYS)
    bi.prices[-1] = BenchmarkPrice(bi.prices[-1].day, bi.prices[-1].price * D("1.2"))
    changed = compare(portfolio(DAYS, flows), bi, "GBP", True)
    assert full.series[:-1] == changed.series[:-1]
    assert full.series[-1] != changed.series[-1]


def test_uniform_price_rescaling_does_not_change_results() -> None:
    """Back-adjusting history by a constant factor must not move any comparison output."""
    p = portfolio(DAYS, {DAYS[0]: 1000.0, DAYS[30]: 200.0})
    a = compare(p, bench_input(DAYS), "GBP", True)
    b = compare(p, bench_input(DAYS, start_price=37.0), "GBP", True)
    assert [x.benchmark_value for x in a.series] == pytest.approx(
        [x.benchmark_value for x in b.series]
    )
    assert get(a, "benchmark_xirr_closing").value == pytest.approx(
        get(b, "benchmark_xirr_closing").value
    )


def test_xirr_is_withheld_when_period_is_too_short() -> None:
    days = weekdays(10)
    c = compare(portfolio(days, {days[0]: 1000.0}), bench_input(days), "GBP", True)
    assert c.status == "available"
    assert get(c, "benchmark_xirr_closing").status == "unavailable"
    assert get(c, "xirr_difference").status == "unavailable"


# --- benchmark data: conversion, provenance, caching ---------------------------------------------


def frame(days: list[date], adj: list[Any], close: list[Any] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {"date": pd.to_datetime(days), "close": close or adj, "adj_close": adj, "dividend": 0.0}
    )


def test_conversion_uses_adjusted_close_and_same_day_fx_and_drops_not_fills() -> None:
    days = weekdays(5, date(2024, 1, 8))
    fx = fx_table(usd_per_eur="1.25", gbp_per_eur="1", days=60)  # 1 USD = 0.8 GBP
    f = frame(days, [100.0, float("nan"), 102.0, -1.0, 104.0], close=[200.0] * 5)
    prices, bad, no_fx = to_account_currency(f, fx, "GBP", "USD")
    assert (bad, no_fx) == (2, 0)
    assert [p.day for p in prices] == [days[0], days[2], days[4]]
    assert prices[0].price == D("100") * D("0.8")  # adj_close, never close: no double counting
    far = frame([date(2030, 1, 7)], [100.0])  # no FX rate within the staleness bound
    assert to_account_currency(far, fx, "GBP", "USD") == ([], 0, 1)


class FakeProvider:
    name = "fake"

    def __init__(self, df: pd.DataFrame | None, currency: str | None = "USD") -> None:
        self.df, self.currency = df, currency
        self.calls: list[tuple[str, date, date]] = []

    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        self.calls.append((symbol, start, end))
        if self.df is None:
            raise ProviderError("boom")
        out = self.df.copy()
        out.attrs["currency"] = self.currency
        return out


def test_fetch_cache_provenance_and_load_roundtrip(tmp_path: Path) -> None:
    days = weekdays(10, date(2024, 1, 8))
    prov = FakeProvider(frame(days, [100.0 + i for i in range(10)]))
    status = fetch_benchmark(tmp_path, prov, days[0], today=days[-1])
    assert status == "ok: 10 rows" and prov.calls[0][0] == "SPY"
    assert prov.calls[0][1] == days[0] - timedelta(days=14)  # only symbol and dates are sent
    meta = json.loads((tmp_path / "benchmarks" / "SPY.meta.json").read_text())
    assert len(meta["sha256"]) == 64 and meta["currency"] == "USD"
    assert "not redistributable" in meta["note"]
    bi = load_benchmark(tmp_path, fx_table(days=60), "GBP")
    assert not bi.problems and len(bi.prices) == 10
    assert bi.source is not None and bi.source.rows_usable == 10
    assert bi.source.sha256 == meta["sha256"]
    assert bi.prices[0].price == D("100") * D("0.8")


def test_load_refuses_tampered_wrong_currency_missing_fx_or_missing_cache(tmp_path: Path) -> None:
    assert "No cached SPY" in load_benchmark(tmp_path, fx_table(), "GBP").problems[0]
    days = weekdays(5, date(2024, 1, 8))
    fetch_benchmark(tmp_path, FakeProvider(frame(days, [1.0] * 5)), days[0], today=days[-1])
    assert any("Exchange rates" in p for p in load_benchmark(tmp_path, None, "GBP").problems)
    csv = tmp_path / "benchmarks" / "SPY.csv"
    csv.write_text(csv.read_text().replace("1.0", "9.0", 1))
    assert any("fingerprint" in p for p in load_benchmark(tmp_path, fx_table(), "GBP").problems)
    other = tmp_path / "o"
    fetch_benchmark(other, FakeProvider(frame(days, [1.0] * 5), "GBp"), days[0], today=days[-1])
    assert any("quote currency" in p for p in load_benchmark(other, fx_table(), "GBP").problems)


def test_fetch_failure_is_reported_not_cached(tmp_path: Path) -> None:
    status = fetch_benchmark(tmp_path, FakeProvider(None), date(2024, 1, 1), today=date(2024, 2, 1))
    assert status.startswith("failed") and not (tmp_path / "benchmarks").exists()

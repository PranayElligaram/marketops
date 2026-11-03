"""XIRR: known answers, signs, multiple roots, non-solutions and invalid input (synthetic)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from marketops.performance.xirr import CashFlow, xirr

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def cf(days: float, amount: float) -> CashFlow:
    return CashFlow(T0 + timedelta(days=days), amount)


def test_single_deposit_one_year_ten_percent() -> None:
    r = xirr([cf(0, -100), cf(365, 110)])
    assert r.status == "ok" and r.rate is not None and math.isclose(r.rate, 0.10, abs_tol=1e-9)
    assert r.warnings == []


def test_loss_gives_negative_rate() -> None:
    r = xirr([cf(0, -100), cf(365, 90)])
    assert r.rate is not None and math.isclose(r.rate, -0.10, abs_tol=1e-9)


def test_rate_zeroes_the_npv_for_multiple_flows_and_ignores_input_order() -> None:
    flows = [cf(0, -1000), cf(100, -500), cf(200, 300), cf(500, 1500)]
    a, b = xirr(flows), xirr(list(reversed(flows)))
    assert a.status == "ok" and a.rate == b.rate and a.rate is not None
    npv = sum(f.amount * (1 + a.rate) ** (-(f.when - T0).days / 365) for f in flows)
    assert abs(npv) < 1e-6


def test_withdrawal_is_a_positive_flow_not_a_loss() -> None:
    loss = xirr([cf(0, -1000), cf(180, 200), cf(400, 700)])  # 900 back for 1000 in
    gain = xirr([cf(0, -1000), cf(180, 200), cf(400, 1000)])  # 1200 back for 1000 in
    assert loss.rate is not None and loss.rate < 0
    assert gain.rate is not None and gain.rate > 0


def test_timing_of_deposits_matters_not_just_the_total() -> None:
    early = xirr([cf(0, -100), cf(100, -100), cf(365, 230)])
    late = xirr([cf(0, -100), cf(300, -100), cf(365, 230)])
    assert early.rate is not None and late.rate is not None and late.rate > early.rate


def test_multiple_roots_are_reported_as_ambiguous_not_as_a_rate() -> None:
    r = xirr([cf(0, -100), cf(365, 230), cf(730, -132)])
    assert r.status == "ambiguous" and r.rate is None
    assert len(r.roots) == 2
    assert math.isclose(r.roots[0], 0.10, abs_tol=1e-6)
    assert math.isclose(r.roots[1], 0.20, abs_tol=1e-6)


def test_no_solution_when_loss_exceeds_search_range() -> None:
    r = xirr([cf(0, -100), cf(365, 0.001)])
    assert r.status == "no_solution" and r.rate is None


def test_zero_terminal_value_is_insufficient_not_a_rate() -> None:
    assert xirr([cf(0, -100), cf(365, 0)]).status == "insufficient"


def test_insufficient_flows() -> None:
    assert xirr([]).status == "insufficient"
    assert xirr([cf(0, -100)]).status == "insufficient"
    assert xirr([cf(0, -100), cf(400, -50)]).status == "insufficient"  # nothing received
    assert xirr([cf(0, 100), cf(400, 50)]).status == "insufficient"  # nothing paid in


def test_short_period_refused_and_sub_year_period_flagged() -> None:
    short = xirr([cf(0, -100), cf(10, 101)])
    assert short.status == "insufficient" and "30" in short.reason
    half = xirr([cf(0, -100), cf(200, 105)])
    assert half.status == "ok" and half.warnings


def test_invalid_dates_and_amounts() -> None:
    naive = CashFlow(datetime(2024, 1, 1), -100)
    assert xirr([naive, cf(400, 110)]).status == "invalid"
    assert xirr([cf(0, float("nan")), cf(400, 110)]).status == "invalid"
    assert xirr([cf(0, float("inf")), cf(400, 110)]).status == "invalid"


def test_same_day_flows_are_insufficient() -> None:
    assert xirr([cf(0, -100), cf(0, 110)]).status == "insufficient"


def test_extreme_span_does_not_crash() -> None:
    r = xirr([cf(0, -100), cf(365 * 400, 5000)])
    assert r.status in ("ok", "invalid", "no_solution")

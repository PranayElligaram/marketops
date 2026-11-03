"""simulate_benchmark: hand-calculated expectations on synthetic data only (no I/O)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal as D

import pytest

from marketops.valuation.benchmark import (
    BenchmarkPoint,
    BenchmarkPrice,
    ExternalFlow,
    simulate_benchmark,
)

TOL = D("1e-12")


def P(day: str, price: str) -> BenchmarkPrice:
    return BenchmarkPrice(date.fromisoformat(day), D(price))


def F(day: str, amount: str) -> ExternalFlow:
    return ExternalFlow(date.fromisoformat(day), D(amount))


def close(a: D, b: str | D) -> bool:
    return abs(a - D(b)) <= TOL


def row(p: BenchmarkPoint) -> tuple[str, D, D, D]:
    return p.day.isoformat(), p.units, p.value, p.net_contributions


# Worked example (also documented in the design note). 2024-01-04 has no price.
W_PRICES = [
    P("2024-01-01", "10"),
    P("2024-01-02", "12"),
    P("2024-01-03", "11"),
    P("2024-01-05", "20"),
    P("2024-01-08", "25"),
]
W_FLOWS = [F("2024-01-01", "1000"), F("2024-01-03", "-220"), F("2024-01-04", "400")]
W_EXPECTED = [  # day, units, value, net contributions
    ("2024-01-01", "100", "1000", "1000"),
    ("2024-01-02", "100", "1200", "1000"),
    ("2024-01-03", "80", "880", "780"),
    ("2024-01-05", "100", "2000", "1180"),
    ("2024-01-08", "100", "2500", "1180"),
]


def test_worked_example() -> None:
    out = simulate_benchmark(W_PRICES, W_FLOWS)
    assert len(out) == len(W_EXPECTED)
    for got, (day, units, value, nc) in zip(out, W_EXPECTED, strict=True):
        assert got.day.isoformat() == day
        assert close(got.units, units) and close(got.value, value)
        assert close(got.net_contributions, nc)


def test_outputs_are_decimal() -> None:
    p = simulate_benchmark(W_PRICES, W_FLOWS)[0]
    assert isinstance(p.units, D) and isinstance(p.value, D) and isinstance(p.net_contributions, D)


def test_empty_inputs() -> None:
    assert simulate_benchmark([], []) == []


def test_no_flows_gives_zero_everywhere() -> None:
    out = simulate_benchmark([P("2024-01-01", "10"), P("2024-01-02", "11")], [])
    assert [row(p)[1:] for p in out] == [(D(0), D(0), D(0))] * 2


def test_value_is_zero_before_first_flow_then_tracks_price() -> None:
    prices = [P("2024-01-01", "10"), P("2024-01-02", "20"), P("2024-01-03", "30")]
    out = simulate_benchmark(prices, [F("2024-01-02", "100")])
    assert [p.value for p in out] == [0, 100, 150]  # bought 5 units at 20, priced 30 next day
    assert [p.units for p in out] == [0, 5, 5]
    assert [p.net_contributions for p in out] == [0, 100, 100]


def test_flow_earns_nothing_on_its_own_day() -> None:
    out = simulate_benchmark([P("2024-01-01", "10")], [F("2024-01-01", "500")])
    assert close(out[0].value, "500") and close(out[0].net_contributions, "500")


def test_fractional_units_and_non_terminating_division() -> None:
    prices = [P("2024-01-01", "3"), P("2024-01-02", "4.5")]
    out = simulate_benchmark(prices, [F("2024-01-01", "100")])
    assert close(out[0].units, D(100) / D(3)) and close(out[0].value, "100")
    assert close(out[1].value, "150")  # 100/3 * 4.5


def test_same_day_flows_are_netted() -> None:
    prices = [P("2024-01-01", "10"), P("2024-01-02", "10")]
    flows = [F("2024-01-01", "300"), F("2024-01-01", "-100"), F("2024-01-01", "50")]
    out = simulate_benchmark(prices, flows)
    assert close(out[0].units, "25") and close(out[0].net_contributions, "250")


def test_flow_order_does_not_matter() -> None:
    assert simulate_benchmark(W_PRICES, list(reversed(W_FLOWS))) == simulate_benchmark(
        W_PRICES, W_FLOWS
    )


def test_zero_amount_flow_is_a_noop() -> None:
    assert simulate_benchmark(W_PRICES, [*W_FLOWS, F("2024-01-02", "0")]) == simulate_benchmark(
        W_PRICES, W_FLOWS
    )


# --- timing: no look-ahead ---------------------------------------------------------------------


def test_flow_on_unpriced_day_executes_at_next_price_not_previous() -> None:
    prices = [P("2024-01-05", "10"), P("2024-01-08", "20")]  # Fri, Mon
    out = simulate_benchmark(prices, [F("2024-01-06", "200")])  # Saturday
    assert out[0].units == 0 and out[0].net_contributions == 0
    assert close(out[1].units, "10") and close(out[1].value, "200")  # bought at Monday's 20


def test_flow_before_first_price_executes_on_first_price_day() -> None:
    out = simulate_benchmark([P("2024-01-05", "10")], [F("2024-01-01", "100")])
    assert close(out[0].units, "10") and close(out[0].net_contributions, "100")


def test_missing_price_day_is_not_interpolated_or_filled() -> None:
    out = simulate_benchmark(W_PRICES, W_FLOWS)
    assert date(2024, 1, 4) not in {p.day for p in out}


def test_no_future_leakage_prefix_is_unchanged_by_later_data() -> None:
    full = simulate_benchmark(W_PRICES, W_FLOWS)
    prefix = simulate_benchmark(W_PRICES[:3], [f for f in W_FLOWS if f.day <= date(2024, 1, 3)])
    assert full[:3] == prefix
    changed = [*W_PRICES[:4], P("2024-01-08", "999")]  # corrupt only the last day's price
    assert simulate_benchmark(changed, W_FLOWS)[:4] == full[:4]


# --- withdrawals and zero holdings -------------------------------------------------------------


def test_full_withdrawal_leaves_exactly_zero_units() -> None:
    prices = [P("2024-01-01", "3"), P("2024-01-02", "6")]
    flows = [F("2024-01-01", "100"), F("2024-01-02", "-200")]  # 100/3 units * 6 = 200
    out = simulate_benchmark(prices, flows)
    assert out[1].units == 0 and out[1].value == 0 and close(out[1].net_contributions, "-100")


def test_reinvest_after_zero_holdings() -> None:
    prices = [P("2024-01-01", "10"), P("2024-01-02", "20"), P("2024-01-03", "40")]
    flows = [F("2024-01-01", "100"), F("2024-01-02", "-200"), F("2024-01-03", "80")]
    out = simulate_benchmark(prices, flows)
    assert [p.units for p in out] == [10, 0, 2]
    assert [p.value for p in out] == [100, 0, 80]


def test_withdrawal_in_excess_of_holdings_raises() -> None:
    prices = [P("2024-01-01", "10"), P("2024-01-02", "10")]
    with pytest.raises(ValueError):
        simulate_benchmark(prices, [F("2024-01-01", "100"), F("2024-01-02", "-100.01")])


def test_withdrawal_before_any_deposit_raises() -> None:
    with pytest.raises(ValueError):
        simulate_benchmark([P("2024-01-01", "10")], [F("2024-01-01", "-1")])


def test_value_never_negative_and_net_contributions_may_be() -> None:
    prices = [P("2024-01-01", "10"), P("2024-01-02", "30")]
    out = simulate_benchmark(prices, [F("2024-01-01", "100"), F("2024-01-02", "-300")])
    assert all(p.value >= 0 and p.units >= 0 for p in out)
    assert close(out[1].net_contributions, "-200")  # profit taken out: contributions below zero


# --- invalid input -----------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["0", "-5", "NaN", "Infinity", "-Infinity"])
def test_bad_price_raises_value_error(bad: str) -> None:
    with pytest.raises(ValueError):
        simulate_benchmark([P("2024-01-01", bad)], [])


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_flow_raises_value_error(bad: str) -> None:
    with pytest.raises(ValueError):
        simulate_benchmark([P("2024-01-01", "10")], [F("2024-01-01", bad)])


def test_duplicate_price_day_raises() -> None:
    with pytest.raises(ValueError):
        simulate_benchmark([P("2024-01-01", "10"), P("2024-01-01", "11")], [])


def test_unsorted_prices_raise() -> None:
    with pytest.raises(ValueError):
        simulate_benchmark([P("2024-01-02", "10"), P("2024-01-01", "11")], [])


def test_flow_after_last_price_day_raises() -> None:
    with pytest.raises(ValueError):
        simulate_benchmark([P("2024-01-01", "10")], [F("2024-01-02", "5")])


def test_flows_without_prices_raise() -> None:
    with pytest.raises(ValueError):
        simulate_benchmark([], [F("2024-01-01", "5")])


def test_float_inputs_are_rejected_not_coerced() -> None:
    with pytest.raises(TypeError):
        simulate_benchmark([BenchmarkPrice(date(2024, 1, 1), 10.0)], [])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        simulate_benchmark(
            [P("2024-01-01", "10")],
            [ExternalFlow(date(2024, 1, 1), 5.0)],  # type: ignore[arg-type]
        )


# --- invariants --------------------------------------------------------------------------------


def test_invariants_on_worked_example() -> None:
    out = simulate_benchmark(W_PRICES, W_FLOWS)
    assert [p.day for p in out] == [p.day for p in W_PRICES]
    for p, q in zip(out, W_PRICES, strict=True):
        assert p.value == p.units * q.price
        assert p.units >= 0 and p.value >= 0
    assert out[-1].net_contributions == sum((f.amount for f in W_FLOWS), D(0))


def test_scaling_flows_scales_value_and_units_linearly() -> None:
    base = simulate_benchmark(W_PRICES, W_FLOWS)
    tripled = simulate_benchmark(W_PRICES, [ExternalFlow(f.day, f.amount * 3) for f in W_FLOWS])
    for a, b in zip(base, tripled, strict=True):
        assert close(b.value, a.value * 3) and close(b.units, a.units * 3)


def test_rescaling_prices_leaves_value_unchanged() -> None:
    """Total-return back-adjustment multiplies all history by a constant: value must not move."""
    base = simulate_benchmark(W_PRICES, W_FLOWS)
    shrunk = [BenchmarkPrice(p.day, p.price * D("0.37")) for p in W_PRICES]
    scaled = simulate_benchmark(shrunk, W_FLOWS)
    for a, b in zip(base, scaled, strict=True):
        assert close(b.value, a.value) and close(b.net_contributions, a.net_contributions)


def test_value_ratio_between_flow_free_days_equals_price_ratio() -> None:
    out = simulate_benchmark(W_PRICES, W_FLOWS)
    assert close(out[1].value / out[0].value, D("12") / D("10"))  # 01-01 -> 01-02, no flow on 01-02
    assert close(out[4].value / out[3].value, D("25") / D("20"))  # 01-05 -> 01-08

"""Holdings reconstruction and reconciliation (synthetic fills only)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from marketops.performance.holdings import reconcile_holdings, reconstruct

NOW = datetime(2025, 1, 1, tzinfo=UTC)


def fill(
    n: int,
    side: str,
    qty: float,
    value: float | None,
    *,
    day: int = 1,
    ticker: str = "AAA_US_EQ",
    cat: str | None = None,
    fid: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "order_id": n,
        "fill_id": fid if fid is not None else n * 10,
        "ticker": ticker,
        "category": cat or side.lower(),
        "side": side,
        "occurred_at": datetime(2024, 3, day, 10, tzinfo=UTC),
        "quantity": Decimal(str(-qty if side == "SELL" else qty)),
        "net_value": None if value is None else Decimal(str(value)),
        "wallet_currency": "GBP",
        "instrument_currency": "USD",
        "fx_rate": Decimal("1.25"),
        "realised_pnl": None,
        "anomaly": None,
        **extra,
    }


def test_buys_and_partial_sell_use_average_cost() -> None:
    rows = [fill(1, "BUY", 4, 100, day=1), fill(2, "BUY", 4, 140, day=2)]
    rows.append(fill(3, "SELL", 2, 70, day=3))
    rec = reconstruct(rows, NOW, "GBP")["AAA_US_EQ"]
    assert rec.quantity == 6 and rec.cost_basis == 180  # 240 * 6/8
    assert rec.realised_pnl == 70 - 60 and rec.buys == 2 and rec.sells == 1


def test_partial_fills_of_one_order_are_summed() -> None:
    rows = [fill(1, "BUY", 1.5, 30, fid=11), fill(1, "BUY", 2.5, 50, fid=12)]
    rec = reconstruct(rows, NOW, "GBP")["AAA_US_EQ"]
    assert rec.quantity == 4 and rec.cost_basis == 80 and rec.multi_fill_orders == 1


def test_full_sale_closes_the_position_exactly() -> None:
    rec = reconstruct([fill(1, "BUY", 2.5, 100), fill(2, "SELL", 2.5, 120, day=2)], NOW, "GBP")
    assert rec["AAA_US_EQ"].quantity == 0 and rec["AAA_US_EQ"].cost_basis == 0


def test_selling_more_than_held_is_flagged_never_clamped() -> None:
    rec = reconstruct([fill(1, "BUY", 1, 10), fill(2, "SELL", 3, 40, day=2)], NOW, "GBP")[
        "AAA_US_EQ"
    ]
    assert rec.quantity == -2 and "oversold_history_incomplete" in rec.flags
    assert rec.realised_pnl is None
    check = reconcile_holdings({"AAA_US_EQ": rec}, [])
    assert check.status == "mismatch" and check.positions[0].status == "not_held"


def test_corporate_action_fill_is_flagged_not_applied() -> None:
    rows = [fill(1, "BUY", 2, 20), fill(2, "BUY", 2, 0, day=2, cat="corporate_action")]
    rec = reconstruct(rows, NOW, "GBP")["AAA_US_EQ"]
    assert rec.quantity == 2 and "corporate_action_fill_not_applied" in rec.flags
    result = reconcile_holdings({"AAA_US_EQ": rec}, [("AAA_US_EQ", Decimal(4), Decimal(20))])
    pos = result.positions[0]
    assert pos.status == "quantity_mismatch" and any("corporate action" in r for r in pos.reasons)


def test_inconsistent_or_incomplete_fills_are_excluded_and_flagged() -> None:
    rows = [
        fill(1, "BUY", 2, 20, anomaly="wallet_currency_differs_from_account"),
        fill(2, "BUY", 1, None, day=2),
        fill(3, "BUY", 1, 10, day=3, wallet_currency="EUR"),
    ]
    rec = reconstruct(rows, NOW, "GBP")["AAA_US_EQ"]
    assert rec.quantity == 0
    assert {
        "excluded_inconsistent_fill",
        "trade_missing_quantity_or_value",
        "fill_currency_differs_from_account",
    } <= set(rec.flags)


def test_missing_fx_rate_is_flagged_but_value_is_kept() -> None:
    rec = reconstruct([fill(1, "BUY", 2, 20, fx_rate=None)], NOW, "GBP")["AAA_US_EQ"]
    assert rec.quantity == 2 and "missing_fx_rate" in rec.flags


def test_fills_after_as_of_cannot_change_an_earlier_state() -> None:
    rows = [fill(1, "BUY", 2, 20, day=1), fill(2, "BUY", 5, 70, day=10)]
    cut = datetime(2024, 3, 5, tzinfo=UTC)
    assert reconstruct(rows, cut, "GBP")["AAA_US_EQ"].quantity == 2
    assert reconstruct(rows, NOW, "GBP")["AAA_US_EQ"].quantity == 7
    assert reconstruct(rows, cut, "GBP") == reconstruct(rows[:1], cut, "GBP")


def test_ordering_is_deterministic_for_identical_timestamps() -> None:
    rows = [fill(2, "SELL", 1, 15), fill(1, "BUY", 2, 20)]
    assert reconstruct(rows, NOW, "GBP") == reconstruct(list(reversed(rows)), NOW, "GBP")
    assert reconstruct(rows, NOW, "GBP")["AAA_US_EQ"].quantity == 1


def test_reconcile_statuses_and_tolerance() -> None:
    rec = reconstruct(
        [
            fill(1, "BUY", 2.5, 100),
            fill(2, "BUY", 1, 10, ticker="CLOSED_US_EQ"),
            fill(3, "SELL", 1, 12, day=2, ticker="CLOSED_US_EQ"),
            fill(4, "BUY", 3, 30, ticker="GONE_US_EQ"),
        ],
        NOW,
        "GBP",
    )
    live = [
        ("AAA_US_EQ", Decimal("2.5000004"), Decimal("100")),  # inside the 1e-6 tolerance
        ("NEW_US_EQ", Decimal(1), Decimal(5)),
    ]
    result = reconcile_holdings(rec, live)
    by = {p.ticker: p for p in result.positions}
    assert by["AAA_US_EQ"].status == "match" and by["AAA_US_EQ"].cost_status == "match"
    assert by["CLOSED_US_EQ"].status == "closed"
    assert by["GONE_US_EQ"].status == "not_held"
    assert by["NEW_US_EQ"].status == "missing_in_history"
    assert result.status == "mismatch" and result.matched == 2 and result.checked == 4
    assert reconcile_holdings(rec, None).status == "not_checked"


def test_quantity_outside_tolerance_mismatches_and_cost_drift_is_graded() -> None:
    rec = reconstruct([fill(1, "BUY", 2.5, 100)], NOW, "GBP")
    off = reconcile_holdings(rec, [("AAA_US_EQ", Decimal("2.500002"), Decimal("100"))])
    assert off.positions[0].status == "quantity_mismatch"
    for broker_cost, expected in (
        (Decimal("100.3"), "match"),
        (Decimal("101.5"), "drift"),
        (Decimal("110"), "mismatch"),
    ):
        r = reconcile_holdings(rec, [("AAA_US_EQ", Decimal("2.5"), broker_cost)])
        assert r.positions[0].cost_status == expected and r.status == "reconciled"

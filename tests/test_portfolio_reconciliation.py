"""Regression tests: broker-reported vs computed portfolio figures (synthetic data only)."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from marketops.portfolio.analytics import summarise_positions
from tests.test_t212_contract import POSITIONS, SUMMARY


def _ov(summary: Any = SUMMARY, rows: Any = POSITIONS) -> Any:
    return summarise_positions(rows, source="trading212", summary=summary)


def _recon(ov: Any) -> dict[str, Any]:
    return {r.metric: r for r in ov.reconciliation}


def test_consistent_summary_reconciles_every_figure() -> None:
    ov = _ov()
    checks = _recon(ov)
    assert set(checks) == {"Investments value", "Investments cost basis", "Unrealised P/L"}
    assert all(c.status == "match" and c.difference == 0 for c in checks.values())
    assert not any("differs" in w for w in ov.data_quality_warnings)
    assert ov.portfolio_value_incl_cash == 4500.0


def test_value_discrepancy_is_reported_as_mismatch_with_signed_difference() -> None:
    s = copy.deepcopy(SUMMARY)
    s["investments"]["currentValue"] = 4100.0
    c = _recon(_ov(s))["Investments value"]
    assert (c.status, c.broker_reported, c.computed, c.difference) == (
        "mismatch",
        4100.0,
        4000.0,
        100.0,
    )


def test_cost_and_pnl_discrepancies_are_detected_independently() -> None:
    s = copy.deepcopy(SUMMARY)
    s["investments"]["totalCost"] = 3000.0
    s["investments"]["unrealizedProfitLoss"] = 700.0
    ov = _ov(s)
    checks = _recon(ov)
    assert checks["Investments value"].status == "match"
    assert checks["Investments cost basis"].status == "mismatch"
    assert checks["Unrealised P/L"].difference == 200.0
    assert any("cost basis" in w for w in ov.data_quality_warnings)
    # allocation and P/L headline figures stay computed from positions, not from the summary
    assert ov.holdings_cost == 3500.0 and ov.unrealised_pnl == 500.0


def test_rounding_sized_differences_do_not_raise_false_alarms() -> None:
    s = copy.deepcopy(SUMMARY)
    s["investments"]["currentValue"] = 4000.03
    assert _recon(_ov(s))["Investments value"].status == "match"


@pytest.mark.parametrize("missing", ["totalCost", "unrealizedProfitLoss"])
def test_missing_optional_summary_field_is_not_reported_not_assumed_zero(missing: str) -> None:
    s = copy.deepcopy(SUMMARY)
    del s["investments"][missing]
    ov = _ov(s)
    assert ov.cash is not None  # unrelated missing field must not remove cash
    gone = [c for c in ov.reconciliation if c.status == "not_reported"]
    assert len(gone) == 1 and gone[0].broker_reported is None and gone[0].difference is None


def test_missing_realised_pnl_keeps_cash_and_reports_none() -> None:
    s = copy.deepcopy(SUMMARY)
    del s["investments"]["realizedProfitLoss"]
    ov = _ov(s)
    assert ov.cash is not None and ov.broker_realised_pnl_all_time is None
    assert ov.portfolio_value_incl_cash == 4500.0


@pytest.mark.parametrize(
    "breakage",
    [
        lambda s: s.pop("totalValue"),
        lambda s: s["investments"].pop("currentValue"),
        lambda s: s["cash"].pop("inPies"),
        lambda s: s.update(cash=None),
        lambda s: s["cash"].update(availableToTrade="abc"),
        lambda s: s["cash"].update(availableToTrade=-1),
        lambda s: s["investments"].update(totalCost=float("nan")),
    ],
)
def test_unusable_summary_removes_cash_and_reconciliation_but_keeps_positions(
    breakage: Any,
) -> None:
    s = copy.deepcopy(SUMMARY)
    breakage(s)
    ov = _ov(s)
    assert ov.cash is None and ov.reconciliation == [] and ov.portfolio_value_incl_cash is None
    assert ov.holdings_value == 4000.0 and len(ov.holdings) == 3
    assert any("unusable" in w for w in ov.data_quality_warnings)


def test_summary_in_another_currency_is_never_reconciled_or_added() -> None:
    ov = _ov(dict(SUMMARY, currency="USD"))
    assert ov.account_currency == "EUR"  # position wallet currency stays authoritative
    assert ov.cash is None and ov.reconciliation == [] and ov.portfolio_value_incl_cash is None


def test_positions_in_mixed_wallet_currencies_fail_closed() -> None:
    rows = copy.deepcopy(POSITIONS)
    rows[1]["walletImpact"]["currency"] = "GBP"
    with pytest.raises(ValueError, match="mixed wallet currencies"):
        _ov(rows=rows)


@pytest.mark.parametrize("field", ["currentValue", "totalCost", "unrealizedProfitLoss", "currency"])
def test_position_missing_wallet_field_fails_closed(field: str) -> None:
    rows = copy.deepcopy(POSITIONS)
    del rows[0]["walletImpact"][field]
    with pytest.raises(ValueError):
        _ov(rows=rows)


def test_position_pnl_that_contradicts_value_minus_cost_is_flagged_not_corrected() -> None:
    rows = copy.deepcopy(POSITIONS)
    rows[0]["walletImpact"]["unrealizedProfitLoss"] = 123.0
    ov = _ov(rows=rows)
    assert any("AAPL_US_EQ" in w and "P/L" in w for w in ov.data_quality_warnings)
    assert next(h for h in ov.holdings if h.ticker == "AAPL_US_EQ").unrealised_pnl == 123.0


def test_total_value_that_excludes_cash_is_labelled_not_double_counted() -> None:
    s = copy.deepcopy(SUMMARY)
    s["totalValue"] = 4000.0
    ov = _ov(s)
    assert ov.portfolio_value_incl_cash is None and ov.cash_pct_of_portfolio is None
    assert ov.broker_reported_total_value == 4000.0  # broker-reported, kept distinct from computed
    assert ov.holdings_value == 4000.0


def test_broker_reported_and_computed_values_are_separate_fields() -> None:
    s = copy.deepcopy(SUMMARY)
    s["totalValue"] = 4510.0
    ov = _ov(s)
    assert ov.broker_reported_total_value == 4510.0
    assert ov.portfolio_value_incl_cash is None  # disagreement: no combined total is asserted
    assert any("does not reconcile" in w for w in ov.data_quality_warnings)


def test_no_summary_means_no_reconciliation_not_a_fabricated_pass() -> None:
    ov = summarise_positions(POSITIONS, source="trading212")
    assert ov.reconciliation == [] and ov.cash is None
    assert any("Account summary unavailable" in w for w in ov.data_quality_warnings)


def test_flat_account_summary_currency_used_when_no_positions() -> None:
    ov = summarise_positions(
        [],
        source="trading212",
        summary=dict(
            SUMMARY,
            totalValue=500.0,
            investments={
                "currentValue": 0.0,
                "realizedProfitLoss": 5.0,
                "totalCost": 0.0,
                "unrealizedProfitLoss": 0.0,
            },
            cash={"availableToTrade": 500.0, "inPies": 0.0, "reservedForOrders": 0.0},
        ),
    )
    assert ov.account_currency == "EUR" and ov.portfolio_value_incl_cash == 500.0
    assert all(c.status == "match" for c in ov.reconciliation)


def test_small_gap_between_separate_requests_is_drift_not_an_alarm() -> None:
    s = copy.deepcopy(SUMMARY)
    s["investments"]["currentValue"] = 4000.15  # 0.004% of value: a price tick between requests
    ov = _ov(s)
    assert _recon(ov)["Investments value"].status == "drift"
    assert not any("Investments value" in w for w in ov.data_quality_warnings)
    s["investments"]["currentValue"] = 4010.0  # 0.25%: beyond any plausible tick
    assert _recon(_ov(s))["Investments value"].status == "mismatch"

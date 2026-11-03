"""Classification and validation of Trading 212 history records (synthetic data only)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from marketops.history.parse import (
    RecordInvalid,
    dividend_category,
    parse_dividend,
    parse_order,
    parse_transaction,
)
from tests.history_fixtures import div, order, tx


@pytest.mark.parametrize(
    ("kind", "category"),
    [
        ("DEPOSIT", "deposit"),
        ("WITHDRAW", "withdrawal"),
        ("FEE", "fee"),
        ("INTEREST_ON_FREE_CASH", "interest"),
        ("LENDING_INTEREST", "interest"),
        ("TRANSFER", "transfer"),
        ("SOMETHING_NEW", "unclassified"),
    ],
)
def test_transaction_classification_never_guesses(kind: str, category: str) -> None:
    rec = parse_transaction(tx("R1", kind, 10.0))
    assert rec.row["category"] == category
    assert rec.row["tx_type"] == kind  # raw type kept verbatim


def test_transfer_is_not_an_external_flow_and_deposit_sign_is_checked() -> None:
    assert parse_transaction(tx("R", "TRANSFER", 10.0)).row["category"] == "transfer"
    assert parse_transaction(tx("R", "DEPOSIT", -5.0)).row["anomaly"] == "deposit_not_positive"
    assert parse_transaction(tx("R", "DEPOSIT", 5.0)).row["anomaly"] is None
    # withdrawals may be reported with either sign; direction comes from the type
    assert parse_transaction(tx("R", "WITHDRAW", -5.0)).row["anomaly"] is None
    assert parse_transaction(tx("R", "WITHDRAW", 5.0)).row["anomaly"] is None


def test_orders_trades_corporate_actions_and_unfilled_are_distinct() -> None:
    assert parse_order(order(1, "BUY")).row["category"] == "buy"
    assert parse_order(order(2, "SELL")).row["category"] == "sell"
    assert parse_order(order(3, fill=False)).row["category"] == "unfilled"
    assert parse_order(order(4, fill_type="STOCK_SPLIT")).row["category"] == "corporate_action"
    assert parse_order(order(5, fill_type="BRAND_NEW")).row["category"] == "unclassified_fill"


def test_order_source_id_is_stable_and_money_is_decimal() -> None:
    row = parse_order(order(7, "BUY", value=-100.1)).row
    assert row["source_id"] == "7:70"
    assert row["net_value"] == Decimal("-100.1") and isinstance(row["quantity"], Decimal)
    assert row["wallet_currency"] == "GBP" and row["instrument_currency"] == "USD"
    assert parse_order(order(7, fill=False)).row["source_id"] == "7:-"


def test_order_fees_only_when_currency_matches_wallet() -> None:
    same = [{"name": "STAMP_DUTY", "currency": "GBP", "quantity": 0.5}]
    other = [{"name": "FINRA_FEE", "currency": "USD", "quantity": 0.1}]
    assert parse_order(order(1, taxes=same)).row["fees_total"] == Decimal("0.5")
    mixed = parse_order(order(1, taxes=same + other)).row
    assert mixed["fees_total"] is None and len(mixed["taxes"]) == 2  # kept, never converted
    assert parse_order(order(1)).row["fees_total"] is None  # no fee data is not zero fees


def test_order_in_other_currency_than_account_is_flagged() -> None:
    assert parse_order(order(1, ccy="EUR"), "GBP").row["anomaly"] == (
        "wallet_currency_differs_from_account"
    )
    assert parse_order(order(1, ccy="GBP"), "GBP").row["anomaly"] is None


def test_missing_optional_values_stay_none_not_inferred() -> None:
    raw = order(1)
    del raw["fill"]["walletImpact"]["fxRate"]
    del raw["fill"]["walletImpact"]["realisedProfitLoss"]
    row = parse_order(raw).row
    assert row["fx_rate"] is None and row["realised_pnl"] is None


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda r: r["order"].pop("id"), "missing:order.id"),
        (lambda r: r["order"].pop("ticker"), "missing:order.ticker"),
        (lambda r: r["fill"].pop("filledAt"), "missing:fill.filledAt"),
        (lambda r: r["fill"].update(filledAt="2024-03-10T09:00:00"), "invalid:fill.filledAt"),
        (lambda r: r["fill"].update(filledAt="yesterday"), "invalid:fill.filledAt"),
        (lambda r: r["fill"].update(price=float("nan")), "invalid:fill.price"),
        (lambda r: r["fill"].update(quantity=True), "invalid:fill.quantity"),
        (lambda r: r["order"].update(side="HOLD"), "invalid:order.side"),
        (lambda r: r.update(order="x"), "missing:order"),
        (lambda r: r["fill"]["walletImpact"].update(taxes=["x"]), "invalid:taxes"),
    ],
)
def test_malformed_orders_are_rejected_with_value_free_reasons(mutate: Any, reason: str) -> None:
    raw = order(1)
    mutate(raw)
    with pytest.raises(RecordInvalid) as exc:
        parse_order(raw)
    assert exc.value.reason == reason


def test_reject_reasons_never_contain_values() -> None:
    raw = tx("R1", "DEPOSIT", 1.0)
    raw["amount"] = "SECRET-123-AMOUNT"
    with pytest.raises(RecordInvalid) as exc:
        parse_transaction(raw)
    assert "SECRET" not in exc.value.reason and "123" not in exc.value.reason


@pytest.mark.parametrize("field", ["reference", "type", "amount", "currency", "dateTime"])
def test_transactions_require_core_fields(field: str) -> None:
    raw = tx("R", "DEPOSIT", 1.0)
    del raw[field]
    with pytest.raises(RecordInvalid):
        parse_transaction(raw)


def test_dividend_categories_separate_dividends_from_interest_and_distributions() -> None:
    assert dividend_category("ORDINARY") == "dividend"
    assert dividend_category("ORDINARY_MANUFACTURED_PAYMENT") == "dividend"
    assert dividend_category("DIVIDENDS_PAID_BY_US_CORPORATIONS") == "dividend"
    assert dividend_category("INTEREST") == "interest"
    assert dividend_category("INTEREST_PAID_BY_US_OBLIGORS") == "interest"
    assert dividend_category("CAPITAL_GAINS") == "other_distribution"
    assert dividend_category("RETURN_OF_CAPITAL") == "other_distribution"
    assert dividend_category("NEVER_SEEN_BEFORE") == "other_distribution"


def test_dividend_fields_currency_and_anomalies() -> None:
    row = parse_dividend(div("D1", 3.2), "GBP").row
    assert row["amount"] == Decimal("3.2") and row["currency"] == "GBP"
    assert row["ticker_currency"] == "USD" and row["gross_per_share"] == Decimal("0.5")
    assert parse_dividend(div("D1", 3.2, ccy="EUR"), "GBP").row["anomaly"] == (
        "currency_differs_from_account"
    )
    assert parse_dividend(div("D1", -1.0), "GBP").row["anomaly"] == "negative_amount"
    raw = div("D1")
    del raw["amountInEuro"], raw["quantity"]
    assert parse_dividend(raw).row["amount_eur"] is None


def test_identical_records_hash_equal_and_changes_differ() -> None:
    a, b = parse_dividend(div("D1", 3.2)), parse_dividend(div("D1", 3.20))
    assert a.content_hash == b.content_hash
    assert a.content_hash != parse_dividend(div("D1", 3.3)).content_hash

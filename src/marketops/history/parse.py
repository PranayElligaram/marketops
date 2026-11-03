"""Validate and classify Trading 212 history records (schemas from docs.trading212.com OpenAPI).

Rules: a record missing its stable identifier, timestamp or a core amount is rejected with a
reason code, never repaired; optional fields stay None when absent; nothing is converted between
currencies; monetary values are Decimal. Reason codes contain field names only, never values.
Classification uses the documented enums; unknown enum values are kept verbatim and classified
"unclassified" so they are visible rather than guessed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

HistoryKind = Literal["orders", "transactions", "dividends"]


class RecordInvalid(Exception):
    """A broker record that cannot be stored; `reason` is a value-free code like 'missing:id'."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------------------------- primitives
def _dec(value: Any, name: str, *, required: bool = False) -> Decimal | None:
    if value is None:
        if required:
            raise RecordInvalid(f"missing:{name}")
        return None
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise RecordInvalid(f"invalid:{name}")
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise RecordInvalid(f"invalid:{name}") from None
    if not number.is_finite() or abs(number) >= Decimal(10) ** 15:
        raise RecordInvalid(f"invalid:{name}")
    return number


def _ts(value: Any, name: str, *, required: bool = False) -> datetime | None:
    if value is None:
        if required:
            raise RecordInvalid(f"missing:{name}")
        return None
    if not isinstance(value, str):
        raise RecordInvalid(f"invalid:{name}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise RecordInvalid(f"invalid:{name}") from None
    if parsed.tzinfo is None:  # a naive timestamp would need an assumed timezone: reject
        raise RecordInvalid(f"invalid:{name}")
    return parsed.astimezone(UTC)


def _str(value: Any, name: str, *, required: bool = False, maxlen: int = 64) -> str | None:
    if value is None or value == "":
        if required:
            raise RecordInvalid(f"missing:{name}")
        return None
    if not isinstance(value, str) or len(value) > maxlen or not value.isprintable():
        raise RecordInvalid(f"invalid:{name}")
    return value


def _int(value: Any, name: str, *, required: bool = False) -> int | None:
    if value is None:
        if required:
            raise RecordInvalid(f"missing:{name}")
        return None
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise RecordInvalid(f"invalid:{name}")
    try:
        return int(value)
    except ValueError:
        raise RecordInvalid(f"invalid:{name}") from None


def _obj(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RecordInvalid(f"invalid:{name}")
    return value


def content_hash(fields: dict[str, Any]) -> str:
    """Stable digest of the normalised record, used to tell a re-delivery from a changed record."""

    def norm(v: Any) -> Any:
        if isinstance(v, Decimal):
            return format(v.normalize(), "f")
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, list):
            return [norm(x) for x in v]
        if isinstance(v, dict):
            return {k: norm(x) for k, x in v.items()}
        return v

    blob = json.dumps(norm(fields), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


# --------------------------------------------------------------------------- records
@dataclass(frozen=True)
class OrderFillRecord:
    source_id: str
    row: dict[str, Any] = field(compare=False)
    content_hash: str = ""


@dataclass(frozen=True)
class CashTransactionRecord:
    source_id: str
    row: dict[str, Any] = field(compare=False)
    content_hash: str = ""


@dataclass(frozen=True)
class DividendRecord:
    source_id: str
    row: dict[str, Any] = field(compare=False)
    content_hash: str = ""


Record = OrderFillRecord | CashTransactionRecord | DividendRecord


# --------------------------------------------------------------------------- orders
def parse_order(raw: dict[str, Any], account_currency: str | None = None) -> OrderFillRecord:
    order = raw.get("order")
    if not isinstance(order, dict):
        raise RecordInvalid("missing:order")
    fill = raw.get("fill")
    if fill is not None and not isinstance(fill, dict):
        raise RecordInvalid("invalid:fill")
    order_id = _int(order.get("id"), "order.id", required=True)
    fill_id = _int(fill.get("id"), "fill.id", required=True) if fill else None
    ticker = _str(order.get("ticker"), "order.ticker", required=True, maxlen=48)
    side = _str(order.get("side"), "order.side", maxlen=8)
    created_at = _ts(order.get("createdAt"), "order.createdAt")
    wallet = _obj(fill.get("walletImpact"), "fill.walletImpact") if fill else {}
    filled_at = _ts(fill.get("filledAt"), "fill.filledAt", required=True) if fill else None
    occurred_at = filled_at or created_at
    if occurred_at is None:
        raise RecordInvalid("missing:order.createdAt")
    fill_type = _str(fill.get("type"), "fill.type", maxlen=40) if fill else None

    if fill is None:
        category = "unfilled"
    elif fill_type == "TRADE":
        if side not in ("BUY", "SELL"):
            raise RecordInvalid("invalid:order.side")
        category = side.lower()
    elif fill_type is None:
        raise RecordInvalid("missing:fill.type")
    elif fill_type in _KNOWN_NON_TRADE_FILLS:
        category = "corporate_action"
    else:
        category = "unclassified_fill"

    taxes: list[dict[str, Any]] = []
    for t in wallet.get("taxes") or []:
        if not isinstance(t, dict):
            raise RecordInvalid("invalid:taxes")
        taxes.append(
            {
                "name": _str(t.get("name"), "taxes.name", required=True, maxlen=40),
                "currency": _str(t.get("currency"), "taxes.currency", maxlen=8),
                "quantity": _dec(t.get("quantity"), "taxes.quantity", required=True),
                "charged_at": _ts(t.get("chargedAt"), "taxes.chargedAt"),
            }
        )
    wallet_ccy = _str(wallet.get("currency"), "walletImpact.currency", maxlen=8)
    fee_ccys = {t["currency"] or wallet_ccy for t in taxes}
    fees_total: Decimal | None = None
    if taxes and len(fee_ccys) == 1 and None not in fee_ccys and fee_ccys == {wallet_ccy}:
        fees_total = sum((t["quantity"] for t in taxes), Decimal(0))

    anomaly: str | None = None
    if fill and category in ("buy", "sell"):
        if wallet_ccy is None or wallet.get("netValue") is None:
            anomaly = "trade_without_wallet_value"
        elif account_currency and wallet_ccy != account_currency:
            anomaly = "wallet_currency_differs_from_account"
    instrument = _obj(order.get("instrument"), "order.instrument")
    row: dict[str, Any] = {
        "source_id": f"{order_id}:{fill_id if fill_id is not None else '-'}",
        "order_id": order_id,
        "fill_id": fill_id,
        "ticker": ticker,
        "category": category,
        "side": side,
        "order_type": _str(order.get("type"), "order.type", maxlen=16),
        "order_status": _str(order.get("status"), "order.status", maxlen=24),
        "fill_type": fill_type,
        "trading_method": _str(fill.get("tradingMethod"), "fill.tradingMethod", maxlen=8)
        if fill
        else None,
        "occurred_at": occurred_at,
        "quantity": _dec(fill.get("quantity"), "fill.quantity") if fill else None,
        "price": _dec(fill.get("price"), "fill.price") if fill else None,
        "order_currency": _str(order.get("currency"), "order.currency", maxlen=8),
        "instrument_currency": _str(instrument.get("currency"), "instrument.currency", maxlen=8),
        "wallet_currency": wallet_ccy,
        "net_value": _dec(wallet.get("netValue"), "walletImpact.netValue"),
        "fx_rate": _dec(wallet.get("fxRate"), "walletImpact.fxRate"),
        "realised_pnl": _dec(wallet.get("realisedProfitLoss"), "walletImpact.realisedProfitLoss"),
        "fees_total": fees_total,
        "taxes": [
            {
                **t,
                "quantity": format(t["quantity"].normalize(), "f"),
                "charged_at": t["charged_at"].isoformat() if t["charged_at"] else None,
            }
            for t in taxes
        ],
        "anomaly": anomaly,
    }
    return OrderFillRecord(row["source_id"], row, content_hash(row))


_KNOWN_NON_TRADE_FILLS = {
    "STOCK_SPLIT",
    "STOCK_DISTRIBUTION",
    "FOP",
    "FOP_CORRECTION",
    "CUSTOM_STOCK_DISTRIBUTION",
    "EQUITY_RIGHTS",
    "SCRIP_STOCK_DIVIDENDS",
    "STOCK_DIVIDENDS",
    "STOCK_ACQUISITION",
    "CASH_AND_STOCK_ACQUISITION",
    "SPIN_OFF",
}


# --------------------------------------------------------------------------- transactions
# TRANSFER is deliberately not a deposit or withdrawal: the API does not document its direction
# or counterparty, so it is kept visible but never counted as an external cash flow.
TRANSACTION_CATEGORIES = {
    "DEPOSIT": "deposit",
    "WITHDRAW": "withdrawal",
    "FEE": "fee",
    "INTEREST_ON_FREE_CASH": "interest",
    "LENDING_INTEREST": "interest",
    "TRANSFER": "transfer",
}


def parse_transaction(raw: dict[str, Any]) -> CashTransactionRecord:
    reference = _str(raw.get("reference"), "reference", required=True)
    tx_type = _str(raw.get("type"), "type", required=True, maxlen=32)
    amount = _dec(raw.get("amount"), "amount", required=True)
    currency = _str(raw.get("currency"), "currency", required=True, maxlen=8)
    occurred_at = _ts(raw.get("dateTime"), "dateTime", required=True)
    assert reference and tx_type and amount is not None and currency and occurred_at
    category = TRANSACTION_CATEGORIES.get(tx_type, "unclassified")
    anomaly = None
    if category == "deposit" and amount <= 0:
        anomaly = "deposit_not_positive"  # direction is defined by type; a non-positive deposit
    elif category == "withdrawal" and amount == 0:  # is inconsistent, not silently re-signed
        anomaly = "withdrawal_zero"
    row = {
        "source_id": reference,
        "tx_type": tx_type,
        "category": category,
        "occurred_at": occurred_at,
        "amount": amount,
        "currency": currency,
        "anomaly": anomaly,
    }
    return CashTransactionRecord(reference, row, content_hash(row))


# --------------------------------------------------------------------------- dividends
def dividend_category(div_type: str) -> str:
    base = div_type.removesuffix("_MANUFACTURED_PAYMENT")
    if base in _DIVIDEND_TYPES:
        return "dividend"
    if base.startswith("INTEREST"):
        return "interest"
    return "other_distribution"


_DIVIDEND_TYPES = {
    "ORDINARY",
    "BONUS",
    "DIVIDEND",
    "DIVIDENDS_PAID_BY_US_CORPORATIONS",
    "DIVIDENDS_PAID_BY_FOREIGN_CORPORATIONS",
}


def parse_dividend(raw: dict[str, Any], account_currency: str | None = None) -> DividendRecord:
    reference = _str(raw.get("reference"), "reference", required=True)
    ticker = _str(raw.get("ticker"), "ticker", required=True, maxlen=48)
    div_type = _str(raw.get("type"), "type", required=True, maxlen=80)
    amount = _dec(raw.get("amount"), "amount", required=True)
    currency = _str(raw.get("currency"), "currency", required=True, maxlen=8)
    paid_on = _ts(raw.get("paidOn"), "paidOn", required=True)
    assert reference and ticker and div_type and amount is not None and currency and paid_on
    anomaly = None
    if amount < 0:
        anomaly = "negative_amount"
    elif account_currency and currency != account_currency:
        anomaly = "currency_differs_from_account"
    instrument = _obj(raw.get("instrument"), "instrument")
    row = {
        "source_id": reference,
        "ticker": ticker,
        "div_type": div_type,
        "category": dividend_category(div_type),
        "occurred_at": paid_on,
        "amount": amount,
        "currency": currency,
        "amount_eur": _dec(raw.get("amountInEuro"), "amountInEuro"),
        "gross_per_share": _dec(raw.get("grossAmountPerShare"), "grossAmountPerShare"),
        "quantity": _dec(raw.get("quantity"), "quantity"),
        "ticker_currency": _str(raw.get("tickerCurrency"), "tickerCurrency", maxlen=8),
        "instrument_currency": _str(instrument.get("currency"), "instrument.currency", maxlen=8),
        "anomaly": anomaly,
    }
    return DividendRecord(reference, row, content_hash(row))


def parse_record(kind: HistoryKind, raw: dict[str, Any], account_currency: str | None) -> Record:
    if kind == "orders":
        return parse_order(raw, account_currency)
    if kind == "transactions":
        return parse_transaction(raw)
    return parse_dividend(raw, account_currency)

"""Portfolio analytics from broker-reported values in the account currency.

Rules: use walletImpact (account currency) for every monetary aggregate, never quote prices
times quantity; report instrument currency only as exposure labels; never invent FX rates,
returns or history. Anything the broker did not supply is reported as unavailable.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, Field

Source = Literal["trading212", "synthetic_example"]

# Descriptive rule-of-thumb thresholds for warnings, not investment advice.
SINGLE_POSITION_WARN_PCT = 25.0
TOP3_WARN_PCT = 60.0
MIN_HOLDINGS_WARN = 5
FOREIGN_CCY_WARN_PCT = 50.0
SUMMARY_MISMATCH_TOLERANCE = 0.02  # absolute, account currency (rounding only)
# Summary and positions are separate requests, so live prices can move between them. A gap up
# to this share of holdings value is reported as "drift" (informational), larger ones as mismatch.
SNAPSHOT_DRIFT_FRACTION = 0.001


class Holding(BaseModel):
    ticker: str
    name: str
    isin: str | None = None
    quantity: float = Field(ge=0)
    currency: str  # wallet (account) currency of the monetary fields below
    instrument_currency: str | None = None
    current_value: float
    total_cost: float
    unrealised_pnl: float
    unrealised_pnl_pct: float | None = None  # None when cost is zero
    fx_impact: float | None = None  # broker-reported; None when not applicable/provided
    weight_pct: float = Field(ge=0, le=100)  # share of open-holdings value


class CashSummary(BaseModel):
    available_to_trade: float
    in_pies: float
    reserved_for_orders: float
    total_cash: float  # sum of the three broker-reported cash components


class Exposure(BaseModel):
    key: str
    value: float
    weight_pct: float


class Concentration(BaseModel):
    holdings_count: int
    largest_ticker: str | None
    largest_weight_pct: float
    top3_weight_pct: float
    herfindahl_index: float
    effective_number_of_holdings: float
    warnings: list[str]


class ReconciliationCheck(BaseModel):
    """Broker-reported figure vs the figure MarketOps computed from positions (same snapshot)."""

    metric: str
    broker_reported: float | None
    computed: float
    difference: float | None
    status: Literal["match", "drift", "mismatch", "not_reported"]


class Performance(BaseModel):
    available: bool
    reason: str


class PortfolioOverview(BaseModel):
    source: Source
    broker_environment: Literal["demo", "live"] | None = None
    account_currency: str
    snapshot_utc: datetime
    holdings: list[Holding]
    holdings_value: float
    holdings_cost: float
    unrealised_pnl: float
    unrealised_pnl_pct: float | None = None
    largest_weight_pct: float
    cash: CashSummary | None = None
    portfolio_value_incl_cash: float | None = None
    cash_pct_of_portfolio: float | None = None
    broker_reported_total_value: float | None = None
    broker_realised_pnl_all_time: float | None = None
    reconciliation: list[ReconciliationCheck] = []
    concentration: Concentration
    currency_exposure: list[Exposure]
    isin_country_exposure: list[Exposure]
    total_fx_impact: float | None = None
    performance: Performance
    data_quality_warnings: list[str]
    note: str


def _num(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid monetary value in broker data ({what})") from exc
    if not isfinite(number):
        raise ValueError("Non-finite monetary value in broker data")
    return number


def _opt_num(value: Any, what: str) -> float | None:
    return None if value is None else _num(value, what)


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def parse_cash(
    summary: Mapping[str, Any], account_currency: str
) -> tuple[CashSummary, dict[str, Any]]:
    """Validate the account summary against the positions' currency; fail closed."""
    if summary.get("currency") != account_currency:
        raise ValueError("Account summary currency differs from position wallet currency")
    cash = summary.get("cash")
    inv = summary.get("investments")
    if not isinstance(cash, dict) or not isinstance(inv, dict):
        raise ValueError("Account summary missing cash or investments")
    parts = [_num(cash[k], k) for k in ("availableToTrade", "inPies", "reservedForOrders")]
    if min(parts) < 0:
        raise ValueError("Negative cash component in broker data")
    # Realised P/L and cost/P&L totals are informational: a missing one must not remove cash.
    extra = {
        "total_value": _num(summary["totalValue"], "totalValue"),
        "inv_value": _num(inv["currentValue"], "investments.currentValue"),
        "realised": _opt_num(inv.get("realizedProfitLoss"), "realizedProfitLoss"),
        "inv_cost": _opt_num(inv.get("totalCost"), "investments.totalCost"),
        "inv_pnl": _opt_num(inv.get("unrealizedProfitLoss"), "investments.unrealizedProfitLoss"),
    }
    return CashSummary(
        available_to_trade=parts[0],
        in_pies=parts[1],
        reserved_for_orders=parts[2],
        total_cash=round(sum(parts), 2),
    ), extra


def _exposures(values: Mapping[str, float]) -> list[Exposure]:
    total = sum(values.values())
    rows = [
        Exposure(key=k, value=round(v, 2), weight_pct=round(100 * v / total, 2) if total else 0)
        for k, v in values.items()
    ]
    return sorted(rows, key=lambda e: (-e.value, e.key))


def concentration_of(holdings: Sequence[Holding]) -> Concentration:
    weights = sorted((h.weight_pct for h in holdings), reverse=True)
    shares = [h.weight_pct / 100 for h in holdings]
    hhi = sum(s * s for s in shares)
    top3 = round(sum(weights[:3]), 2)
    largest = max(holdings, key=lambda h: h.current_value, default=None)
    warnings: list[str] = []
    if holdings and weights[0] > SINGLE_POSITION_WARN_PCT:
        warnings.append(
            f"{largest.ticker if largest else 'A position'} is {weights[0]:.1f}% of holdings "
            f"(above the {SINGLE_POSITION_WARN_PCT:.0f}% single-position guide)."
        )
    if len(holdings) > 3 and top3 > TOP3_WARN_PCT:
        warnings.append(
            f"Top 3 positions are {top3:.1f}% of holdings (guide: {TOP3_WARN_PCT:.0f}%)."
        )
    if 0 < len(holdings) < MIN_HOLDINGS_WARN:
        warnings.append(
            f"Only {len(holdings)} holding(s): little diversification across positions."
        )
    return Concentration(
        holdings_count=len(holdings),
        largest_ticker=largest.ticker if largest else None,
        largest_weight_pct=weights[0] if weights else 0,
        top3_weight_pct=top3,
        herfindahl_index=round(hhi, 4),
        effective_number_of_holdings=round(1 / hhi, 2) if hhi else 0,
        warnings=warnings,
    )


def summarise_positions(
    rows: Sequence[Mapping[str, Any]],
    source: Source,
    summary: Mapping[str, Any] | None = None,
) -> PortfolioOverview:
    """Fail closed on inconsistent currencies or missing monetary values in positions.

    A missing/inconsistent account summary only removes the cash section (with a warning).
    """
    parsed: list[dict[str, Any]] = []
    currencies: set[str] = set()
    for row in rows:
        instrument = row.get("instrument")
        wallet = row.get("walletImpact")
        if not isinstance(instrument, dict) or not isinstance(wallet, dict):
            raise ValueError("Missing instrument or walletImpact in broker data")
        currency = wallet.get("currency")
        if not isinstance(currency, str) or not currency:
            raise ValueError("Missing account currency from broker data")
        currencies.add(currency)
        ticker = instrument.get("ticker")
        if not isinstance(ticker, str) or not ticker:
            raise ValueError("Missing position ticker")
        try:
            quantity = float(row["quantity"])
            value = float(wallet["currentValue"])
            cost = float(wallet["totalCost"])
            pnl = float(wallet["unrealizedProfitLoss"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("Invalid monetary value in broker data") from exc
        if not all(isfinite(n) for n in (quantity, value, cost, pnl)):
            raise ValueError("Non-finite monetary value in broker data")
        if quantity < 0 or value < 0:
            raise ValueError("Negative quantity or market value in broker data")
        fx = wallet.get("fxImpact")
        fx_impact = _num(fx, "fxImpact") if fx is not None else None
        parsed.append(
            {
                "ticker": ticker,
                "name": str(instrument.get("name") or ticker),
                "isin": _opt_str(instrument.get("isin")),
                "quantity": quantity,
                "currency": currency,
                "instrument_currency": _opt_str(instrument.get("currency")),
                "value": value,
                "cost": cost,
                "pnl": pnl,
                "fx": fx_impact,
            }
        )
    if len(currencies) > 1:
        raise ValueError("Positions contain mixed wallet currencies")
    total = sum(p["value"] for p in parsed)
    holdings = [
        Holding(
            ticker=p["ticker"],
            name=p["name"],
            isin=p["isin"],
            quantity=p["quantity"],
            currency=p["currency"],
            instrument_currency=p["instrument_currency"],
            current_value=p["value"],
            total_cost=p["cost"],
            unrealised_pnl=p["pnl"],
            unrealised_pnl_pct=round(100 * p["pnl"] / p["cost"], 2) if p["cost"] > 0 else None,
            fx_impact=p["fx"],
            weight_pct=round(100 * p["value"] / total, 2) if total else 0,
        )
        for p in parsed
    ]
    holdings.sort(key=lambda h: h.current_value, reverse=True)
    account_ccy = next(iter(currencies), None)
    if account_ccy is None and summary is not None:
        # No open positions: the account summary is the only currency authority.
        summary_ccy = summary.get("currency")
        account_ccy = summary_ccy if isinstance(summary_ccy, str) and summary_ccy else None
    cost = sum(h.total_cost for h in holdings)
    pnl = sum(h.unrealised_pnl for h in holdings)

    warnings: list[str] = []
    for h in holdings:
        if abs((h.current_value - h.total_cost) - h.unrealised_pnl) > SUMMARY_MISMATCH_TOLERANCE:
            warnings.append(f"{h.ticker}: broker P/L differs from value minus cost.")

    by_ccy: dict[str, float] = defaultdict(float)
    for h in holdings:
        by_ccy[h.instrument_currency or "Unknown"] += h.current_value
    foreign = sum(v for k, v in by_ccy.items() if k not in (account_ccy, "Unknown"))
    if total and 100 * foreign / total > FOREIGN_CCY_WARN_PCT:
        warnings.append(
            f"{100 * foreign / total:.0f}% of holdings are priced in a currency other than "
            f"{account_ccy}; their account-currency value moves with FX rates."
        )
    by_isin: dict[str, float] = defaultdict(float)
    for h in holdings:
        by_isin[h.isin[:2].upper() if h.isin else "Unknown"] += h.current_value
    fx_values = [h.fx_impact for h in holdings if h.fx_impact is not None]

    cash: CashSummary | None = None
    extra: dict[str, Any] = {}
    if summary is not None and account_ccy is not None:
        try:
            cash, extra = parse_cash(summary, account_ccy)
        except (ValueError, KeyError):
            cash, extra = None, {}
            warnings.append("Account summary was unusable; cash figures are unavailable.")
    elif summary is None and source == "trading212":
        warnings.append("Account summary unavailable; cash figures are not shown.")

    # The API docs do not state whether totalValue includes cash or whether the three cash
    # fields are additive, so a combined total is only shown when the broker's own figures
    # agree: totalValue == investments.currentValue + the three cash fields (same snapshot).
    incl: float | None = None
    if cash:
        tol = SUMMARY_MISMATCH_TOLERANCE * max(1, len(holdings))
        if abs(extra["total_value"] - (extra["inv_value"] + cash.total_cash)) <= tol:
            incl = round(total + cash.total_cash, 2)
        elif abs(extra["total_value"] - extra["inv_value"]) <= tol and cash.total_cash > tol:
            warnings.append(
                "Broker totalValue equals investments only, so cash is not included in it; "
                "a combined total is not shown because cash fields were not verified additive."
            )
        else:
            warnings.append(
                "Broker totalValue does not reconcile with investments plus cash; "
                "a combined total is not shown."
            )
    reconciliation: list[ReconciliationCheck] = []
    if extra:
        tol = SUMMARY_MISMATCH_TOLERANCE * max(1, len(holdings))
        for metric, key, computed in (
            ("Investments value", "inv_value", total),
            ("Investments cost basis", "inv_cost", cost),
            ("Unrealised P/L", "inv_pnl", pnl),
        ):
            reported = extra.get(key)
            diff = None if reported is None else round(reported - computed, 2)
            reconciliation.append(
                ReconciliationCheck(
                    metric=metric,
                    broker_reported=reported,
                    computed=round(computed, 2),
                    difference=diff,
                    status="not_reported"
                    if reported is None
                    else "match"
                    if abs(reported - computed) <= tol
                    else "drift"
                    if abs(reported - computed) <= SNAPSHOT_DRIFT_FRACTION * max(total, 1.0)
                    else "mismatch",
                )
            )
        for r in reconciliation:
            if r.status == "mismatch":
                warnings.append(
                    f"{r.metric}: broker summary differs from the sum of positions by "
                    f"{r.difference:+.2f} (summary and positions are separate requests)."
                )
    return PortfolioOverview(
        source=source,
        account_currency=account_ccy or "N/A",
        snapshot_utc=datetime.now(UTC),
        holdings=holdings,
        holdings_value=round(total, 2),
        holdings_cost=round(cost, 2),
        unrealised_pnl=round(pnl, 2),
        unrealised_pnl_pct=round(100 * pnl / cost, 2) if cost > 0 else None,
        largest_weight_pct=max((h.weight_pct for h in holdings), default=0),
        cash=cash,
        portfolio_value_incl_cash=incl,
        cash_pct_of_portfolio=round(100 * cash.total_cash / incl, 2) if cash and incl else None,
        broker_reported_total_value=extra.get("total_value"),
        broker_realised_pnl_all_time=extra.get("realised"),
        reconciliation=reconciliation,
        concentration=concentration_of(holdings),
        currency_exposure=_exposures(by_ccy),
        isin_country_exposure=_exposures(by_isin),
        total_fx_impact=round(sum(fx_values), 2) if fx_values else None,
        performance=Performance(
            available=False,
            reason=(
                "Time-weighted/money-weighted returns need verified portfolio valuations at each "
                "cash-flow date; transaction history alone does not provide them (see Portfolio "
                "activity for what is verified). Current holdings alone cannot give historical "
                "performance, so none is shown."
            ),
        ),
        data_quality_warnings=warnings,
        note=(
            "Illustrative synthetic data, not a real account."
            if source == "synthetic_example"
            else "Open positions from Trading 212 walletImpact (account currency). Unrealised "
            "P/L is since purchase, not a period return. ISIN prefix is the issuer's "
            "registration country, not economic exposure."
        ),
    )


EXAMPLE_POSITIONS: list[dict[str, Any]] = [
    {
        "instrument": {
            "ticker": "EXAMPLE_A",
            "name": "Synthetic Company A",
            "currency": "USD",
            "isin": "US0000000001",
        },
        "quantity": 10,
        "walletImpact": {
            "currency": "GBP",
            "currentValue": 600,
            "totalCost": 560,
            "unrealizedProfitLoss": 40,
            "fxImpact": -5,
        },
    },
    {
        "instrument": {
            "ticker": "EXAMPLE_B",
            "name": "Synthetic ETF B",
            "currency": "GBP",
            "isin": "IE0000000002",
        },
        "quantity": 20,
        "walletImpact": {
            "currency": "GBP",
            "currentValue": 400,
            "totalCost": 420,
            "unrealizedProfitLoss": -20,
            "fxImpact": None,
        },
    },
]

EXAMPLE_ACCOUNT_SUMMARY: dict[str, Any] = {
    "id": 0,
    "currency": "GBP",
    "totalValue": 1250,
    "cash": {"availableToTrade": 240, "inPies": 0, "reservedForOrders": 10},
    "investments": {
        "currentValue": 1000,
        "realizedProfitLoss": 15,
        "totalCost": 980,
        "unrealizedProfitLoss": 20,
    },
}

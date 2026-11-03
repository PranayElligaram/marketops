"""Holdings reconstructed from executed fills, reconciled against a live positions snapshot.

Nothing is repaired: a quantity that does not match is reported with reasons, oversold positions
stay negative, and corporate-action fills are flagged rather than interpreted. Only fills at or
before `as_of` are used, so a later record can never change an earlier state.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel

QTY_TOLERANCE = Decimal("0.000001")  # shares
COST_DRIFT_FRACTION = Decimal("0.005")  # 0.5 % of broker cost: informational, not gating
COST_MISMATCH_FRACTION = Decimal("0.02")


@dataclass
class _State:
    qty: Decimal = Decimal(0)
    cost: Decimal = Decimal(0)
    realised: Decimal = Decimal(0)
    reported_realised: Decimal = Decimal(0)
    buys: int = 0
    sells: int = 0
    order_fills: dict[int, int] = field(default_factory=dict)
    flags: set[str] = field(default_factory=set)
    first: datetime | None = None
    last: datetime | None = None
    realised_valid: bool = True


class Reconstructed(BaseModel):
    ticker: str
    quantity: float
    cost_basis: float  # average-cost method, wallet currency
    realised_pnl: float | None  # None when the position was ever oversold
    broker_reported_realised_pnl: float
    buys: int
    sells: int
    multi_fill_orders: int
    first_fill_utc: datetime | None
    last_fill_utc: datetime | None
    flags: list[str]


def _signed_qty(row: Mapping[str, Any]) -> Decimal | None:
    q = row.get("quantity")
    if q is None or Decimal(q) == 0:
        return None
    return abs(Decimal(q)) * (1 if row["category"] == "buy" else -1)


def reconstruct(
    fills: Iterable[Mapping[str, Any]], as_of: datetime, account_currency: str | None
) -> dict[str, Reconstructed]:
    """`fills` are broker_order_fill rows. Order: time, then order id, then fill id (stable)."""
    rows = sorted(
        (r for r in fills if r["occurred_at"] <= as_of),
        key=lambda r: (r["occurred_at"], r["order_id"], r["fill_id"] or 0),
    )
    states: dict[str, _State] = {}
    for r in rows:
        cat = r["category"]
        if cat == "unfilled":
            continue
        st = states.setdefault(r["ticker"], _State())
        if r.get("anomaly"):
            st.flags.add("excluded_inconsistent_fill")
            continue
        if cat == "corporate_action":
            st.flags.add("corporate_action_fill_not_applied")
            continue
        if cat == "unclassified_fill":
            st.flags.add("unclassified_fill_not_applied")
            continue
        if cat not in ("buy", "sell"):
            st.flags.add("unknown_fill_category")
            continue
        qty = _signed_qty(r)
        value = None if r.get("net_value") is None else abs(Decimal(r["net_value"]))
        if qty is None or value is None:
            st.flags.add("trade_missing_quantity_or_value")
            continue
        if account_currency and r.get("wallet_currency") != account_currency:
            st.flags.add("fill_currency_differs_from_account")
            continue
        if (
            r.get("instrument_currency")
            and r["instrument_currency"] != r.get("wallet_currency")
            and r.get("fx_rate") is None
        ):
            st.flags.add("missing_fx_rate")
        st.first = st.first or r["occurred_at"]
        st.last = r["occurred_at"]
        st.order_fills[r["order_id"]] = st.order_fills.get(r["order_id"], 0) + 1
        if r.get("realised_pnl") is not None:
            st.reported_realised += Decimal(r["realised_pnl"])
        if qty > 0:
            st.buys += 1
            st.qty += qty
            st.cost += value
        else:
            st.sells += 1
            sold = -qty
            if st.qty <= 0 or sold > st.qty + QTY_TOLERANCE:
                st.flags.add("oversold_history_incomplete")
                st.realised_valid = False
                st.qty -= sold
                continue
            removed = st.cost * sold / st.qty
            st.realised += value - removed
            st.cost -= removed
            st.qty -= sold
        if abs(st.qty) <= QTY_TOLERANCE:
            st.qty, st.cost = Decimal(0), Decimal(0)
    return {
        t: Reconstructed(
            ticker=t,
            quantity=float(s.qty),
            cost_basis=float(s.cost),
            realised_pnl=float(s.realised) if s.realised_valid else None,
            broker_reported_realised_pnl=float(s.reported_realised),
            buys=s.buys,
            sells=s.sells,
            multi_fill_orders=sum(1 for n in s.order_fills.values() if n > 1),
            first_fill_utc=s.first,
            last_fill_utc=s.last,
            flags=sorted(s.flags),
        )
        for t, s in sorted(states.items())
    }


PositionStatus = Literal["match", "closed", "quantity_mismatch", "missing_in_history", "not_held"]
CostStatus = Literal["match", "drift", "mismatch", "not_checked"]
Overall = Literal["reconciled", "mismatch", "not_checked"]


class PositionCheck(BaseModel):
    ticker: str
    status: PositionStatus
    reconstructed_quantity: float
    broker_quantity: float | None
    quantity_difference: float | None
    cost_status: CostStatus
    reconstructed_cost: float | None
    broker_cost: float | None
    reasons: list[str]


class HoldingsReconciliation(BaseModel):
    status: Overall
    tolerance_shares: float
    checked: int
    matched: int
    positions: list[PositionCheck]


def _cost_status(recon: Decimal, broker: Decimal | None) -> CostStatus:
    if broker is None or broker == 0:
        return "not_checked"
    gap = abs(recon - broker) / abs(broker)
    if gap <= COST_DRIFT_FRACTION:
        return "match"
    return "drift" if gap <= COST_MISMATCH_FRACTION else "mismatch"


def reconcile_holdings(
    reconstructed: Mapping[str, Reconstructed],
    broker_positions: Sequence[tuple[str, Decimal, Decimal]] | None,
) -> HoldingsReconciliation:
    """`broker_positions`: (ticker, quantity, total cost) from the live snapshot, or None."""
    if broker_positions is None:
        return HoldingsReconciliation(
            status="not_checked",
            tolerance_shares=float(QTY_TOLERANCE),
            checked=0,
            matched=0,
            positions=[],
        )
    live = {t: (q, c) for t, q, c in broker_positions}
    checks: list[PositionCheck] = []
    for ticker in sorted(set(reconstructed) | set(live)):
        rec = reconstructed.get(ticker)
        bq, bc = live.get(ticker, (None, None))
        rq = Decimal(str(rec.quantity)) if rec else Decimal(0)
        reasons: list[str] = []
        flags = rec.flags if rec else []
        diff = None if bq is None else rq - bq
        status: PositionStatus
        if rec is None:
            status = "missing_in_history"
            reasons.append("Held at the broker but no executed fill exists in the stored history.")
        elif bq is None:
            if abs(rq) <= QTY_TOLERANCE:
                status = "closed"
            else:
                status = "not_held"
                reasons.append("History implies shares that the broker does not report as held.")
        else:
            status = "match" if abs(rq - bq) <= QTY_TOLERANCE else "quantity_mismatch"
            if rq < 0:
                reasons.append("Reconstructed quantity is negative: earlier buys are missing.")
        if status in ("missing_in_history", "quantity_mismatch", "not_held"):
            reasons += [f"Flag: {f.replace('_', ' ')}." for f in flags]
            if not flags:
                reasons.append(
                    "No corporate-action or data flag explains it; possible causes are history "
                    "before the first record, a ticker change, or an unsupported event."
                )
        else:
            reasons += [f"Note: {f.replace('_', ' ')}." for f in flags]
        cost_status: CostStatus = (
            _cost_status(Decimal(str(rec.cost_basis)), bc)
            if rec and bc is not None
            else "not_checked"
        )
        checks.append(
            PositionCheck(
                ticker=ticker,
                status=status,
                reconstructed_quantity=float(rq),
                broker_quantity=None if bq is None else float(bq),
                quantity_difference=None if diff is None else float(diff),
                cost_status=cost_status,
                reconstructed_cost=rec.cost_basis if rec else None,
                broker_cost=None if bc is None else float(bc),
                reasons=reasons,
            )
        )
    bad = [c for c in checks if c.status not in ("match", "closed")]
    return HoldingsReconciliation(
        status="mismatch" if bad else "reconciled",
        tolerance_shares=float(QTY_TOLERANCE),
        checked=len(checks),
        matched=len(checks) - len(bad),
        positions=checks,
    )

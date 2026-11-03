"""Historical performance report: data audit, holdings reconstruction, cash-ledger check, metrics.

A metric is `available` only when every input it needs reconciles; otherwise it is `partial`
(shown with blockers) or `unavailable` (with the specific missing input). Deposits are never
profit, buys/sells/dividends/interest/fees are internal to the account, and only records at or
before `as_of` (the broker snapshot time) are used, so nothing later can leak into a valuation.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel
from sqlalchemy import Table, select
from sqlalchemy.engine import Engine

from marketops.history.read import Coverage, KindCoverage
from marketops.performance.holdings import (
    HoldingsReconciliation,
    Reconstructed,
    reconcile_holdings,
    reconstruct,
)
from marketops.performance.xirr import CashFlow, XirrResult, xirr
from marketops.persistence.models import (
    broker_cash_transaction,
    broker_dividend,
    broker_order_fill,
)
from marketops.portfolio.analytics import PortfolioOverview

PAGE_SIZE = 50
LEDGER_ABS_TOLERANCE = Decimal("1.00")  # account currency
LEDGER_REL_TOLERANCE = Decimal("0.0001")  # of gross purchase value
REALISED_ABS_TOLERANCE = Decimal("0.50")
REALISED_REL_TOLERANCE = Decimal("0.005")  # of sale proceeds

MetricStatus = Literal["available", "partial", "unavailable"]


class Check(BaseModel):
    name: str
    ok: bool | None  # None = could not be evaluated
    detail: str


class KindAudit(BaseModel):
    kind: str
    records: int
    earliest_utc: str | None
    latest_utc: str | None
    types: dict[str, int]
    amount_signs: dict[str, dict[str, int]]  # type -> {"positive": n, "negative": n}
    currencies: list[str]
    checks: list[Check]


class DataAudit(BaseModel):
    kinds: list[KindAudit]
    inception: Check
    issues: list[str]


class LedgerCheck(BaseModel):
    status: Literal["match", "mismatch", "not_checked"]
    tolerance: float
    difference: float | None  # reconstructed minus broker cash
    reconstructed_cash: float | None
    broker_cash: float | None
    components: dict[str, float]
    reasons: list[str]


class FlowPoint(BaseModel):
    date: str
    kind: Literal["deposit", "withdrawal", "terminal_value"]
    amount: float  # investor-signed: paid in negative, received positive


class NetInvestedPoint(BaseModel):
    date: str
    net_invested: float


class Metric(BaseModel):
    name: str
    status: MetricStatus
    value: dict[str, float] | None = None
    unit: str | None = None
    basis: str
    blockers: list[str] = []
    cross_check: str | None = None


class HistoricalPerformance(BaseModel):
    as_of_utc: str
    environment: str
    account_currency: str | None
    snapshot_available: bool
    overall_status: Literal["verified", "partial", "unverified"]
    data_audit: DataAudit
    holdings: HoldingsReconciliation
    reconstructed_positions: list[Reconstructed]
    cash_ledger: LedgerCheck
    xirr: XirrResult | None
    external_flows: list[FlowPoint]
    net_invested_series: list[NetInvestedPoint]
    metrics: list[Metric]
    notes: list[str]


# ------------------------------------------------------------------ loading
def _utc(v: datetime) -> datetime:
    return v.replace(tzinfo=UTC) if v.tzinfo is None else v.astimezone(UTC)


def _dec(v: Any) -> Decimal:
    return Decimal(str(v))


def _load(engine: Engine, table: Table, env: str, key: str) -> list[dict[str, Any]]:
    with engine.connect() as c:
        rows = (
            c.execute(select(table).where(table.c.environment == env, table.c.account_key == key))
            .mappings()
            .all()
        )
    out = [dict(r) for r in rows]
    for r in out:
        r["occurred_at"] = _utc(r["occurred_at"])
    return out


def _iso(v: datetime | None) -> str | None:
    return v.isoformat() if v else None


# ------------------------------------------------------------------ audit
def _audit_kind(
    kind: str, rows: list[dict[str, Any]], cov: Coverage, type_col: str | None
) -> KindAudit:
    kc = next(k for k in cov.kinds if k.kind == kind)
    ids = [r["source_id"] for r in rows]
    times = [r["occurred_at"] for r in rows]
    types: Counter[str] = Counter(str(r[type_col]) if type_col else r["category"] for r in rows)
    signs: dict[str, dict[str, int]] = {}
    if kind != "orders" and type_col:
        for r in rows:
            s = signs.setdefault(str(r[type_col]), {"positive": 0, "negative": 0})
            s["positive" if r["amount"] > 0 else "negative"] += int(r["amount"] != 0)
    checks = [
        Check(
            name="unique_identifiers",
            ok=len(set(ids)) == len(ids),
            detail=f"{len(set(ids))} distinct identifiers for {len(ids)} stored records.",
        ),
        Check(
            name="full_pass_reached_end",
            ok=kc.backfill_complete_utc is not None,
            detail="A full pass followed pagination to its final page."
            if kc.backfill_complete_utc
            else "No full pass has followed pagination to its final page.",
        ),
        Check(
            name="no_rejected_or_conflicting",
            ok=kc.rejected_records == 0 and kc.conflicting_records == 0,
            detail=f"{kc.rejected_records} rejected, {kc.conflicting_records} conflicting.",
        ),
        Check(
            name="no_inconsistent_records",
            ok=kc.inconsistent_records == 0,
            detail=f"{kc.inconsistent_records} stored record(s) flagged inconsistent.",
        ),
    ]
    ccys = sorted(
        {str(r[c]) for r in rows for c in ("currency", "wallet_currency") if c in r and r[c]}
    )
    return KindAudit(
        kind=kind,
        records=len(rows),
        earliest_utc=_iso(min(times)) if times else None,
        latest_utc=_iso(max(times)) if times else None,
        types=dict(sorted(types.items())),
        amount_signs=signs,
        currencies=ccys,
        checks=checks,
    )


def _page_checks(engine: Engine, env: str, key: str, audits: list[KindAudit]) -> None:
    """Compare the last finished full pass's counters with the stored row counts."""
    from marketops.persistence.models import broker_sync_run as runs

    with engine.connect() as c:
        for a in audits:
            run = (
                c.execute(
                    select(runs)
                    .where(
                        runs.c.environment == env,
                        runs.c.account_key == key,
                        runs.c.kind == a.kind,
                        runs.c.mode == "full",
                        runs.c.status == "complete",
                        runs.c.reached_end.is_(True),
                    )
                    .order_by(runs.c.id.desc())
                    .limit(1)
                )
                .mappings()
                .first()
            )
            if run is None:
                a.checks.append(
                    Check(name="pagination_arithmetic", ok=None, detail="No full pass.")
                )
                continue
            seen = int(run["records_seen"])
            expected = max(1, math.ceil(seen / PAGE_SIZE))
            pages = int(run["pages_fetched"])
            accounted = sum(
                int(run[k])
                for k in (
                    "records_inserted",
                    "records_duplicate",
                    "records_rejected",
                    "records_conflicting",
                )
            )
            a.checks.append(
                Check(
                    name="pagination_arithmetic",
                    ok=pages >= expected and accounted == seen,
                    detail=f"{pages} page(s) for {seen} record(s) (at least {expected} needed at "
                    f"{PAGE_SIZE} per page); {accounted} of {seen} accounted for.",
                )
            )
            a.checks.append(
                Check(
                    name="stored_equals_imported",
                    ok=a.records == int(run["records_inserted"]),
                    detail=f"{a.records} stored vs {int(run['records_inserted'])} imported "
                    "in that pass (later incremental runs may add records).",
                )
            )


# ------------------------------------------------------------------ ledger
def _cash_ledger(
    tx: list[dict[str, Any]],
    divs: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    broker_cash: Decimal | None,
    ccy: str | None,
) -> LedgerCheck:
    def total(rows: list[dict[str, Any]], cat: str, signed: bool) -> Decimal:
        return sum(
            (_dec(r["amount"]) if signed else abs(_dec(r["amount"])))
            for r in rows
            if r["category"] == cat and not r["anomaly"] and r["currency"] == ccy
        ) or Decimal(0)

    trades = [r for r in fills if r["category"] in ("buy", "sell") and not r["anomaly"]]
    buys = sum((abs(_dec(r["net_value"])) for r in trades if r["category"] == "buy"), Decimal(0))
    sells = sum((abs(_dec(r["net_value"])) for r in trades if r["category"] == "sell"), Decimal(0))
    dividends = sum(
        (_dec(r["amount"]) for r in divs if not r["anomaly"] and r["currency"] == ccy), Decimal(0)
    )
    comp = {
        "deposits": total(tx, "deposit", False),
        "withdrawals": -total(tx, "withdrawal", False),
        "cash_fees": -total(tx, "fee", False),
        "interest": total(tx, "interest", True),
        "dividends_and_distributions": dividends,
        "purchases": -buys,
        "sales": sells,
    }
    reconstructed = sum(comp.values(), Decimal(0))
    reasons: list[str] = []
    if sum(1 for r in tx if r["category"] in ("transfer", "unclassified")):
        reasons.append(
            "TRANSFER or unclassified cash transactions exist and are not in the ledger."
        )
    if any(r["category"] in ("buy", "sell") and r["anomaly"] for r in fills):
        reasons.append("Some trade fills are flagged inconsistent and are not in the ledger.")
    if any(r["category"] == "corporate_action" for r in fills):
        reasons.append("Corporate-action fills exist; their cash effect is not modelled.")
    tolerance = LEDGER_ABS_TOLERANCE + LEDGER_REL_TOLERANCE * buys
    if broker_cash is None or ccy is None:
        reasons.append("No live cash balance to compare against.")
        status: Literal["match", "mismatch", "not_checked"] = "not_checked"
        diff = None
    else:
        diff = reconstructed - broker_cash
        status = "match" if abs(diff) <= tolerance and not reasons else "mismatch"
        if abs(diff) > tolerance:
            reasons.append(
                "Cash implied by the stored history differs from the broker balance by more "
                "than the tolerance: events may be missing, or order fees/taxes may not be "
                "reflected in the recorded net values."
            )
    return LedgerCheck(
        status=status,
        tolerance=float(tolerance),
        difference=None if diff is None else float(diff),
        reconstructed_cash=float(reconstructed) if broker_cash is not None else None,
        broker_cash=None if broker_cash is None else float(broker_cash),
        components={k: float(v) for k, v in comp.items()},
        reasons=reasons,
    )


# ------------------------------------------------------------------ main
def build_report(
    engine: Engine,
    environment: str,
    account_key: str,
    cov: Coverage,
    snapshot: PortfolioOverview | None,
    now: datetime | None = None,
) -> HistoricalPerformance:
    ccy = cov.account_currency
    as_of = _utc(snapshot.snapshot_utc) if snapshot else _utc(now or datetime.now(UTC))
    # Records after the snapshot are ignored so the valuation never sees the future.
    tx = [r for r in _load(engine, broker_cash_transaction, environment, account_key)]
    divs = [r for r in _load(engine, broker_dividend, environment, account_key)]
    fills = [r for r in _load(engine, broker_order_fill, environment, account_key)]
    all_rows = {"transactions": tx, "dividends": divs, "orders": fills}
    tx, divs, fills = ([r for r in rows if r["occurred_at"] <= as_of] for rows in (tx, divs, fills))

    audits = [
        _audit_kind("orders", fills, cov, "category"),
        _audit_kind("transactions", tx, cov, "tx_type"),
        _audit_kind("dividends", divs, cov, "div_type"),
    ]
    _page_checks(engine, environment, account_key, audits)
    for a in audits:
        a.checks.append(
            Check(
                name="records_after_snapshot_ignored",
                ok=True,
                detail=f"{len(all_rows[a.kind]) - a.records} record(s) dated after the snapshot "
                "were excluded.",
            )
        )
    issues = [f"{k.kind}: {i}" for k in cov.kinds for i in k.issues]
    by_kind: dict[str, KindCoverage] = {k.kind: k for k in cov.kinds}

    # --- holdings
    recon = reconstruct(fills, as_of, ccy)
    live_positions = (
        [(h.ticker, _dec(h.quantity), _dec(h.total_cost)) for h in snapshot.holdings]
        if snapshot
        else None
    )
    holdings = reconcile_holdings(recon, live_positions)

    # --- cash ledger
    broker_cash = _dec(snapshot.cash.total_cash) if snapshot and snapshot.cash is not None else None
    ledger = _cash_ledger(tx, divs, fills, broker_cash, ccy)

    # --- structural gates
    def clean(kind: str) -> list[str]:
        k = by_kind[kind]
        out: list[str] = []
        if k.state != "complete":
            out.append(f"{kind} history is not complete (state: {k.state}).")
        else:
            out += k.issues
        return out

    tx_blockers = clean("transactions")
    if not ccy:
        tx_blockers.append("Account currency is unknown.")
    bad_ccy = {r["currency"] for r in tx if r["category"] in ("deposit", "withdrawal")} - {ccy}
    if bad_ccy:
        tx_blockers.append("External flows exist in a currency other than the account currency.")
    n_transfer = sum(1 for r in tx if r["category"] == "transfer")
    n_unclass = sum(1 for r in tx if r["category"] == "unclassified")
    if n_transfer:
        tx_blockers.append(f"{n_transfer} TRANSFER record(s) with undocumented direction.")
    if n_unclass:
        tx_blockers.append(f"{n_unclass} transaction(s) of an unsupported type.")

    deposits = [
        r for r in tx if r["category"] == "deposit" and not r["anomaly"] and r["currency"] == ccy
    ]
    withdrawals = [
        r for r in tx if r["category"] == "withdrawal" and not r["anomaly"] and r["currency"] == ccy
    ]
    dep_total = sum((abs(_dec(r["amount"])) for r in deposits), Decimal(0))
    wd_total = sum((abs(_dec(r["amount"])) for r in withdrawals), Decimal(0))

    # --- inception evidence
    first_tx = min(tx, key=lambda r: r["occurred_at"]) if tx else None
    inception_ok = bool(
        first_tx
        and first_tx["category"] == "deposit"
        and ledger.status == "match"
        and holdings.status == "reconciled"
    )
    inception = Check(
        name="history_starts_at_account_inception",
        ok=True if inception_ok else (None if ledger.status == "not_checked" else False),
        detail=(
            "The earliest record is a deposit and the stored history reproduces the current "
            "cash balance and positions from a zero opening balance. This is consistent with, "
            "but does not prove, that history starts at account opening."
            if inception_ok
            else "Not established: the API does not state the account opening date, and the "
            "stored history does not reproduce the current cash and positions."
        ),
    )

    flows_ok = not tx_blockers
    flows_verified = flows_ok and ledger.status == "match"
    flow_status: MetricStatus = (
        "unavailable" if not tx else "available" if flows_verified else "partial"
    )
    flow_blockers = list(tx_blockers)
    if flows_ok and ledger.status != "match":
        flow_blockers.append(
            "Completeness is unverified: the cash ledger does not reproduce the broker balance."
            if ledger.status == "mismatch"
            else "No live cash balance was available to verify completeness."
        )
    metrics: list[Metric] = []
    metrics.append(
        Metric(
            name="total_external_deposits",
            status=flow_status,
            value={"amount": float(dep_total), "count": float(len(deposits))} if tx else None,
            unit=ccy,
            basis="Sum of DEPOSIT records (magnitudes) in the account currency.",
            blockers=flow_blockers,
        )
    )
    metrics.append(
        Metric(
            name="total_external_withdrawals",
            status=flow_status,
            value={"amount": float(wd_total), "count": float(len(withdrawals))} if tx else None,
            unit=ccy,
            basis="Sum of WITHDRAW records (magnitudes) in the account currency.",
            blockers=flow_blockers,
        )
    )
    metrics.append(
        Metric(
            name="net_invested_capital",
            status=flow_status,
            value={"amount": float(dep_total - wd_total)} if tx else None,
            unit=ccy,
            basis="Deposits minus withdrawals. Buys, sells, dividends, interest and fees are "
            "internal to the account and are not external flows.",
            blockers=flow_blockers,
        )
    )

    # --- dividends and interest
    div_blockers = clean("dividends")
    div_dep = [r for r in divs if not r["anomaly"]]
    if {r["currency"] for r in div_dep} - {ccy}:
        div_blockers.append("Dividends exist in a non-account currency; no FX is applied.")
    div_status: MetricStatus = (
        "unavailable"
        if by_kind["dividends"].state == "never_synced"
        else "available"
        if not div_blockers
        else "partial"
    )

    def dsum(cat: str) -> float:
        return float(sum((_dec(r["amount"]) for r in div_dep if r["category"] == cat), Decimal(0)))

    metrics.append(
        Metric(
            name="dividends_received",
            status=div_status,
            value={
                "amount": dsum("dividend"),
                "count": float(sum(r["category"] == "dividend" for r in div_dep)),
            }
            if divs
            else None,
            unit=ccy,
            basis="Dividend-type payouts as credited by the broker (net/gross not documented). "
            "Capital distributions are excluded.",
            blockers=div_blockers,
        )
    )
    int_status: MetricStatus = (
        "unavailable"
        if by_kind["transactions"].state == "never_synced"
        else "available"
        if not clean("transactions")
        else "partial"
    )
    cash_interest = sum(
        (_dec(r["amount"]) for r in tx if r["category"] == "interest" and not r["anomaly"]),
        Decimal(0),
    )
    cash_fees = sum(
        (abs(_dec(r["amount"])) for r in tx if r["category"] == "fee" and not r["anomaly"]),
        Decimal(0),
    )
    metrics.append(
        Metric(
            name="interest_received",
            status=int_status,
            value={"cash_interest": float(cash_interest), "dividend_interest": dsum("interest")}
            if tx
            else None,
            unit=ccy,
            basis="INTEREST_ON_FREE_CASH / LENDING_INTEREST transactions plus interest-type "
            "dividend records.",
            blockers=clean("transactions"),
        )
    )
    metrics.append(
        Metric(
            name="account_fees",
            status=int_status,
            value={"cash_fees": float(cash_fees)} if tx else None,
            unit=ccy,
            basis="FEE transactions. Per-order fees/taxes are inside trade net values where "
            "the broker includes them.",
            blockers=clean("transactions"),
        )
    )

    # --- realised P/L
    ord_blockers = clean("orders")
    rec_sum = [r.realised_pnl for r in recon.values()]
    recon_realised = (
        None if any(v is None for v in rec_sum) else sum(v for v in rec_sum if v is not None)
    )
    reported = sum((r.broker_reported_realised_pnl for r in recon.values()), 0.0)
    broker_all_time = snapshot.broker_realised_pnl_all_time if snapshot else None
    cross = None
    if broker_all_time is None:
        ord_blockers.append("No live all-time realised P/L to reconcile against.")
    else:
        tol = 0.01 + 0.005 * max(1, sum(r.sells + r.buys for r in recon.values()))
        if abs(reported - broker_all_time) > tol:
            ord_blockers.append(
                "Sum of per-fill realised P/L does not equal the broker's all-time figure."
            )
    if holdings.status != "reconciled":
        ord_blockers.append("Holdings do not reconcile to the broker, so fills may be incomplete.")
    if recon_realised is not None:
        sale_proceeds = sum(
            (
                abs(_dec(r["net_value"]))
                for r in fills
                if r["category"] == "sell" and r["net_value"] is not None
            ),
            Decimal(0),
        )
        gap = abs(Decimal(str(recon_realised)) - Decimal(str(reported)))
        ok = gap <= REALISED_ABS_TOLERANCE + REALISED_REL_TOLERANCE * sale_proceeds
        cross = (
            "Average-cost reconstruction agrees with the broker-reported per-fill figures."
            if ok
            else "Average-cost reconstruction differs from the broker-reported per-fill figures "
            "(the broker may treat fees, FX or cost method differently); use the broker figure."
        )
    else:
        cross = "Average-cost reconstruction is not possible because a position was oversold."
    metrics.append(
        Metric(
            name="realised_pnl",
            status="unavailable"
            if by_kind["orders"].state == "never_synced"
            else "available"
            if not ord_blockers
            else "partial",
            value={"broker_reported_sum_of_fills": reported}
            | (
                {"reconstructed_average_cost": recon_realised} if recon_realised is not None else {}
            ),
            unit=ccy,
            basis="Broker-reported realised P/L summed over fills, reconciled to the live "
            "account figure. Excludes dividends, interest and fees booked outside orders.",
            blockers=ord_blockers,
            cross_check=cross,
        )
    )

    # --- unrealised P/L (snapshot, tied to reconciled holdings)
    unreal_blockers: list[str] = []
    if snapshot is None:
        unreal_blockers.append("No live positions snapshot.")
    elif holdings.status != "reconciled":
        unreal_blockers.append("Reconstructed holdings do not reconcile to the broker positions.")
    metrics.append(
        Metric(
            name="unrealised_pnl",
            status="unavailable"
            if snapshot is None
            else "available"
            if not unreal_blockers
            else "partial",
            value={"amount": snapshot.unrealised_pnl} if snapshot else None,
            unit=ccy,
            basis="Broker-reported unrealised P/L for the current positions at the snapshot time. "
            "A point-in-time figure, not a return.",
            blockers=unreal_blockers,
        )
    )

    # --- valuation-dependent metrics
    val_blockers = list(tx_blockers)
    for kind in ("orders", "dividends"):
        val_blockers += clean(kind)
    if holdings.status != "reconciled":
        val_blockers.append(
            "Reconstructed holdings do not reconcile to the live positions."
            if holdings.status == "mismatch"
            else "Holdings were not checked against live positions."
        )
    if ledger.status != "match":
        val_blockers.append(
            "The cash ledger does not reproduce the broker cash balance."
            if ledger.status == "mismatch"
            else "The cash ledger was not checked against a live cash balance."
        )
    if snapshot is None or snapshot.portfolio_value_incl_cash is None:
        val_blockers.append(
            "No verified terminal valuation (broker total value did not reconcile)."
        )
    elif snapshot.account_currency != ccy:
        val_blockers.append("Snapshot currency differs from the history currency.")
    if not dep_total:
        val_blockers.append("No external deposits are recorded.")
    val_blockers = list(dict.fromkeys(val_blockers))

    flow_points: list[FlowPoint] = []
    result: XirrResult | None = None
    series: list[NetInvestedPoint] = []
    if flows_ok and tx:
        running = Decimal(0)
        for r in sorted(deposits + withdrawals, key=lambda r: r["occurred_at"]):
            sign = 1 if r["category"] == "deposit" else -1
            running += sign * abs(_dec(r["amount"]))
            series.append(
                NetInvestedPoint(
                    date=r["occurred_at"].date().isoformat(), net_invested=float(running)
                )
            )
            flow_points.append(
                FlowPoint(
                    date=r["occurred_at"].isoformat(),
                    kind="deposit" if sign == 1 else "withdrawal",
                    amount=float(-sign * abs(_dec(r["amount"]))),
                )
            )
    if not val_blockers and snapshot and snapshot.portfolio_value_incl_cash is not None:
        flow_points.append(
            FlowPoint(
                date=as_of.isoformat(),
                kind="terminal_value",
                amount=snapshot.portfolio_value_incl_cash,
            )
        )
        result = xirr([CashFlow(datetime.fromisoformat(f.date), f.amount) for f in flow_points])

    terminal = snapshot.portfolio_value_incl_cash if snapshot else None
    gain_value: dict[str, float] | None = None
    if not val_blockers and terminal is not None:
        gain_value = {
            "gain": float(_dec(terminal) + wd_total - dep_total),
            "terminal_value": terminal,
        }
    metrics.append(
        Metric(
            name="net_gain_on_invested_capital",
            status="available" if gain_value else "unavailable",
            value=gain_value,
            unit=ccy,
            basis="Verified portfolio value (positions + cash) plus withdrawals minus deposits. "
            "Includes price changes, dividends, interest, fees and FX; deposits are not profit.",
            blockers=val_blockers,
        )
    )
    xirr_value: dict[str, float] | None = None
    xirr_blockers = list(val_blockers)
    xirr_status: MetricStatus = "unavailable"
    if result is not None:
        if result.status == "ok" and result.rate is not None:
            xirr_status = "available"
            xirr_value = {"annualised_rate": result.rate, "period_days": result.period_days or 0.0}
        else:
            xirr_blockers.append(f"XIRR is not defined ({result.status}): {result.reason}")
            xirr_status = "partial" if result.status == "ambiguous" else "unavailable"
    metrics.append(
        Metric(
            name="money_weighted_return_xirr",
            status=xirr_status,
            value=xirr_value,
            unit="per year",
            basis="Internal rate of return of dated deposits/withdrawals and the verified terminal "
            "value. Sensitive to the timing of deposits; not comparable to a time-weighted return.",
            blockers=xirr_blockers,
        )
    )
    unavailable = {
        "time_weighted_return": "Needs the portfolio value just before every external flow. No "
        "historical valuations exist: only the current snapshot is available.",
        "portfolio_equity_curve": "Needs daily positions x historical prices x FX. No historical "
        "prices for the instruments held, no FX history and no corporate-action records are "
        "stored, so a curve would be invented.",
        "total_return_pct": "Needs a time-weighted return or a defined invested base; "
        "net gain over net deposits ignores when money was added. Use the XIRR above.",
        "benchmark_comparison": "Needs a verified portfolio return series on the same dates and "
        "currency as the benchmark.",
    }
    for name, why in unavailable.items():
        metrics.append(
            Metric(name=name, status="unavailable", basis="Not calculated.", blockers=[why])
        )

    statuses = {
        m.status
        for m in metrics
        if m.name
        in (
            "net_invested_capital",
            "realised_pnl",
            "unrealised_pnl",
            "net_gain_on_invested_capital",
        )
    }
    overall: Literal["verified", "partial", "unverified"] = (
        "verified"
        if statuses == {"available"} and holdings.status == "reconciled"
        else "unverified"
        if snapshot is None or holdings.status != "reconciled" or ledger.status != "match"
        else "partial"
    )
    return HistoricalPerformance(
        as_of_utc=as_of.isoformat(),
        environment=environment,
        account_currency=ccy,
        snapshot_available=snapshot is not None,
        overall_status=overall,
        data_audit=DataAudit(kinds=audits, inception=inception, issues=issues),
        holdings=holdings,
        reconstructed_positions=list(recon.values()),
        cash_ledger=ledger,
        xirr=result,
        external_flows=flow_points,
        net_invested_series=series,
        metrics=metrics,
        notes=[
            "All figures are in the account currency exactly as recorded by the broker; "
            "MarketOps applies no FX conversion.",
            "Only records dated at or before the snapshot time are used.",
        ],
    )

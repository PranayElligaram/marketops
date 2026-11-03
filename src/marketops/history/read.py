"""Read models over stored history: coverage, listings, totals, performance readiness.

Everything here reads only PostgreSQL (never the broker). Totals are exact sums of records the
broker typed explicitly, grouped by currency; nothing is converted between currencies, and
records flagged inconsistent are excluded from totals and counted instead.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, time
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel
from sqlalchemy import Table, and_, func, select
from sqlalchemy.engine import Engine

from marketops.broker.t212 import HistoryKind
from marketops.history.sync import KINDS, TABLES
from marketops.persistence.models import (
    broker_cash_transaction,
    broker_dividend,
    broker_order_fill,
    broker_sync_run,
)

State = Literal["never_synced", "running", "complete", "stale", "partial"]


def _utc(value: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; everything stored is UTC."""
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _iso(value: datetime | None) -> str | None:
    v = _utc(value)
    return v.isoformat() if v else None


def _f(value: Decimal | float | None) -> float | None:
    return None if value is None else float(value)


# ---------------------------------------------------------------- models
class KindCoverage(BaseModel):
    kind: HistoryKind
    state: State
    last_run_status: str | None = None
    last_run_mode: str | None = None
    last_run_started_utc: str | None = None
    last_success_utc: str | None = None
    backfill_complete_utc: str | None = None
    last_error_code: str | None = None
    stored_records: int = 0
    earliest_record_utc: str | None = None
    latest_record_utc: str | None = None
    rejected_records: int = 0
    conflicting_records: int = 0
    inconsistent_records: int = 0
    issues: list[str] = []


class Coverage(BaseModel):
    environment: str
    account_currency: str | None
    kinds: list[KindCoverage]
    sync_running: bool


class CurrencyTotal(BaseModel):
    currency: str
    amount: float
    count: int


class MonthlyTotal(BaseModel):
    month: str
    currency: str
    amount: float


class ActivitySummary(BaseModel):
    date_from: date | None
    date_to: date | None
    account_currency: str | None
    deposits: list[CurrencyTotal]
    withdrawals: list[CurrencyTotal]
    cash_fees: list[CurrencyTotal]
    interest: list[CurrencyTotal]
    transfers_count: int
    unclassified_transactions: int
    dividends: list[CurrencyTotal]
    dividends_by_month: list[MonthlyTotal]
    dividend_interest: list[CurrencyTotal]
    other_distributions: list[CurrencyTotal]
    buy_count: int
    sell_count: int
    buy_value: list[CurrencyTotal]
    sell_value: list[CurrencyTotal]
    order_fees: list[CurrencyTotal]
    realised_pnl_reported: list[CurrencyTotal]
    corporate_action_count: int
    unfilled_order_count: int
    excluded_inconsistent: int
    note: str


class MetricReadiness(BaseModel):
    metric: str
    status: Literal["available", "partial", "unavailable"]
    value: dict[str, float] | None = None
    basis: str
    blockers: list[str] = []


# ---------------------------------------------------------------- scope & coverage
def latest_account_key(engine: Engine, environment: str) -> str | None:
    with engine.connect() as c:
        row = c.execute(
            select(broker_sync_run.c.account_key)
            .where(broker_sync_run.c.environment == environment)
            .order_by(broker_sync_run.c.id.desc())
            .limit(1)
        ).first()
    return str(row[0]) if row else None


def coverage(engine: Engine, environment: str, account_key: str, running: bool) -> Coverage:
    kinds: list[KindCoverage] = []
    account_ccy: str | None = None
    with engine.connect() as c:
        for kind in KINDS:
            runs = (
                c.execute(
                    select(broker_sync_run)
                    .where(
                        broker_sync_run.c.environment == environment,
                        broker_sync_run.c.account_key == account_key,
                        broker_sync_run.c.kind == kind,
                    )
                    .order_by(broker_sync_run.c.id.desc())
                )
                .mappings()
                .all()
            )
            table = TABLES[kind]
            scope = and_(table.c.environment == environment, table.c.account_key == account_key)
            agg: Any = c.execute(
                select(
                    func.count(), func.min(table.c.occurred_at), func.max(table.c.occurred_at)
                ).where(scope)
            ).one()
            count, first, last = agg
            inconsistent = c.execute(
                select(func.count()).where(scope, table.c.anomaly.is_not(None))
            ).scalar_one()
            kc = KindCoverage(
                kind=kind,
                state="never_synced",
                stored_records=int(count or 0),
                earliest_record_utc=_iso(first),
                latest_record_utc=_iso(last),
                inconsistent_records=int(inconsistent),
            )
            if runs:
                latest = runs[0]
                account_ccy = account_ccy or latest["account_currency"]
                backfill = next(
                    (
                        r
                        for r in runs
                        if r["mode"] == "full" and r["status"] == "complete" and r["reached_end"]
                    ),
                    None,
                )
                success = next((r for r in runs if r["status"] == "complete"), None)
                since = [r for r in runs if backfill is None or r["id"] >= backfill["id"]]
                kc.last_run_status = latest["status"]
                kc.last_run_mode = latest["mode"]
                kc.last_run_started_utc = _iso(latest["started_utc"])
                kc.last_success_utc = _iso(success["finished_utc"]) if success else None
                kc.backfill_complete_utc = _iso(backfill["finished_utc"]) if backfill else None
                kc.last_error_code = latest["error_code"]
                kc.rejected_records = sum(r["records_rejected"] for r in since)
                kc.conflicting_records = sum(r["records_conflicting"] for r in since)
                if latest["status"] == "running":
                    kc.state = "running"
                elif backfill is None:
                    kc.state = "partial"
                elif latest["status"] == "complete":
                    kc.state = "complete"
                else:
                    kc.state = "stale"
                kc.issues = _issues(kc)
            kinds.append(kc)
    return Coverage(
        environment=environment, account_currency=account_ccy, kinds=kinds, sync_running=running
    )


def _issues(kc: KindCoverage) -> list[str]:
    out: list[str] = []
    if kc.state == "partial":
        out.append("No full pass has finished: history may be incomplete. Run the sync again.")
    if kc.state == "stale":
        out.append(
            f"The latest update did not finish ({kc.last_error_code or 'unknown'}); "
            "data is complete only up to the last good sync."
        )
    if kc.rejected_records:
        out.append(f"{kc.rejected_records} record(s) were rejected as malformed and are missing.")
    if kc.conflicting_records:
        out.append(f"{kc.conflicting_records} record(s) changed after first import (kept first).")
    if kc.inconsistent_records:
        out.append(f"{kc.inconsistent_records} stored record(s) are inconsistent and excluded.")
    return out


# ---------------------------------------------------------------- listings
def _window(table: Table, d_from: date | None, d_to: date | None) -> list[Any]:
    conds: list[Any] = []
    if d_from:
        conds.append(table.c.occurred_at >= datetime.combine(d_from, time.min, UTC))
    if d_to:
        conds.append(table.c.occurred_at <= datetime.combine(d_to, time.max, UTC))
    return conds


def list_rows(
    engine: Engine,
    kind: HistoryKind,
    environment: str,
    account_key: str,
    d_from: date | None = None,
    d_to: date | None = None,
    limit: int = 200,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    table = TABLES[kind]
    where = [
        table.c.environment == environment,
        table.c.account_key == account_key,
        *_window(table, d_from, d_to),
    ]
    with engine.connect() as c:
        total = int(c.execute(select(func.count()).where(*where)).scalar_one())
        rows = (
            c.execute(
                select(table)
                .where(*where)
                .order_by(table.c.occurred_at.desc(), table.c.id.desc())
                .limit(limit)
                .offset(offset)
            )
            .mappings()
            .all()
        )
    return [_public_row(dict(r)) for r in rows], total


_HIDDEN = {"id", "environment", "account_key", "content_hash", "sync_run_id", "first_seen_utc"}


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key in _HIDDEN:
            continue
        if isinstance(value, datetime):
            out[key] = _iso(value)
        elif isinstance(value, Decimal):
            out[key] = float(value)
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------- totals
def _sum_by_currency(
    rows: Iterable[Any], amount: Callable[[Any], Decimal | None], ccy: Callable[[Any], str | None]
) -> list[CurrencyTotal]:
    sums: dict[str, Decimal] = defaultdict(Decimal)
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        value, code = amount(r), ccy(r)
        if value is None or not code:
            continue
        sums[code] += value
        counts[code] += 1
    return [
        CurrencyTotal(currency=k, amount=round(float(v), 2), count=counts[k])
        for k, v in sorted(sums.items())
    ]


def summarise_activity(
    engine: Engine,
    environment: str,
    account_key: str,
    d_from: date | None = None,
    d_to: date | None = None,
    account_currency: str | None = None,
) -> ActivitySummary:
    def fetch(table: Table) -> list[Any]:
        where = [
            table.c.environment == environment,
            table.c.account_key == account_key,
            *_window(table, d_from, d_to),
        ]
        with engine.connect() as c:
            return list(c.execute(select(table).where(*where)).mappings().all())

    tx_all, div_all, ord_all = (
        fetch(broker_cash_transaction),
        fetch(broker_dividend),
        fetch(broker_order_fill),
    )
    tx = [r for r in tx_all if not r["anomaly"]]
    div = [r for r in div_all if not r["anomaly"]]
    orders = [r for r in ord_all if not r["anomaly"]]

    def by_cat(rows: list[Any], cat: str) -> list[Any]:
        return [r for r in rows if r["category"] == cat]

    def magnitude(r: Any) -> Decimal:
        return Decimal(abs(r["amount"]))

    month_sums: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    for r in by_cat(div, "dividend"):
        month_sums[(_utc(r["occurred_at"]).strftime("%Y-%m"), r["currency"])] += r["amount"]  # type: ignore[union-attr]
    trades = [r for r in orders if r["category"] in ("buy", "sell")]
    fills = [r for r in orders if r["fees_total"] is not None]
    return ActivitySummary(
        date_from=d_from,
        date_to=d_to,
        account_currency=account_currency,
        deposits=_sum_by_currency(by_cat(tx, "deposit"), magnitude, lambda r: r["currency"]),
        withdrawals=_sum_by_currency(by_cat(tx, "withdrawal"), magnitude, lambda r: r["currency"]),
        cash_fees=_sum_by_currency(by_cat(tx, "fee"), magnitude, lambda r: r["currency"]),
        interest=_sum_by_currency(
            by_cat(tx, "interest"), lambda r: r["amount"], lambda r: r["currency"]
        ),
        transfers_count=len(by_cat(tx, "transfer")),
        unclassified_transactions=len(by_cat(tx, "unclassified")),
        dividends=_sum_by_currency(
            by_cat(div, "dividend"), lambda r: r["amount"], lambda r: r["currency"]
        ),
        dividends_by_month=[
            MonthlyTotal(month=m, currency=c, amount=round(float(v), 2))
            for (m, c), v in sorted(month_sums.items())
        ],
        dividend_interest=_sum_by_currency(
            by_cat(div, "interest"), lambda r: r["amount"], lambda r: r["currency"]
        ),
        other_distributions=_sum_by_currency(
            by_cat(div, "other_distribution"), lambda r: r["amount"], lambda r: r["currency"]
        ),
        buy_count=len(by_cat(trades, "buy")),
        sell_count=len(by_cat(trades, "sell")),
        buy_value=_sum_by_currency(
            by_cat(trades, "buy"),
            lambda r: abs(r["net_value"]) if r["net_value"] is not None else None,
            lambda r: r["wallet_currency"],
        ),
        sell_value=_sum_by_currency(
            by_cat(trades, "sell"),
            lambda r: abs(r["net_value"]) if r["net_value"] is not None else None,
            lambda r: r["wallet_currency"],
        ),
        order_fees=_sum_by_currency(
            fills, lambda r: r["fees_total"], lambda r: r["wallet_currency"]
        ),
        realised_pnl_reported=_sum_by_currency(
            trades, lambda r: r["realised_pnl"], lambda r: r["wallet_currency"]
        ),
        corporate_action_count=len(by_cat(orders, "corporate_action")),
        unfilled_order_count=len(by_cat(orders, "unfilled")),
        excluded_inconsistent=sum(
            1 for rows in (tx_all, div_all, ord_all) for r in rows if r["anomaly"]
        ),
        note=(
            "Sums of broker-reported records grouped by currency; no currency conversion. "
            "Deposit/withdrawal direction comes from the broker's record type; amounts are shown "
            "as magnitudes. TRANSFER records are not counted as deposits or withdrawals."
        ),
    )


# ---------------------------------------------------------------- performance readiness
_CURVE_BLOCKER = (
    "No historical portfolio valuations: the API returns current positions only, and "
    "transaction history alone does not establish daily portfolio value."
)


def performance_readiness(
    engine: Engine, environment: str, account_key: str, cov: Coverage
) -> list[MetricReadiness]:
    """Which historical-performance inputs are verified now; everything else is 'unavailable'."""
    by_kind = {k.kind: k for k in cov.kinds}
    ccy = cov.account_currency
    full = summarise_activity(engine, environment, account_key, account_currency=ccy)

    def gate(kind: HistoryKind) -> list[str]:
        k = by_kind[kind]
        out = []
        if k.state != "complete":
            out.append(f"{kind} history is not complete (state: {k.state}).")
        out += k.issues if k.state == "complete" else []
        return out

    out: list[MetricReadiness] = []

    # --- net external cash flows
    blockers = gate("transactions")
    if not ccy:
        blockers.append("Account currency is unknown.")
    flow_ccys = {t.currency for t in full.deposits + full.withdrawals}
    if ccy and flow_ccys - {ccy}:
        blockers.append("Flows exist in a currency other than the account currency; no FX applied.")
    if full.transfers_count:
        blockers.append(
            f"{full.transfers_count} TRANSFER record(s): the API does not state their direction "
            "or whether they are external, so they are not counted."
        )
    if full.unclassified_transactions:
        blockers.append(f"{full.unclassified_transactions} transaction(s) of an unknown type.")
    dep = sum(t.amount for t in full.deposits if t.currency == ccy)
    wd = sum(t.amount for t in full.withdrawals if t.currency == ccy)
    out.append(
        MetricReadiness(
            metric="net_external_flows",
            status="unavailable"
            if by_kind["transactions"].state == "never_synced"
            else "available"
            if not blockers
            else "partial",
            value={
                "deposits": round(dep, 2),
                "withdrawals": round(wd, 2),
                "net": round(dep - wd, 2),
            }
            if ccy and by_kind["transactions"].stored_records
            else None,
            basis="Deposits minus withdrawals in the account currency, from explicitly typed "
            "DEPOSIT/WITHDRAW records as returned by the API. The API cannot prove that history "
            "starts at account opening; check the earliest record date.",
            blockers=blockers,
        )
    )

    # --- dividends
    blockers = gate("dividends")
    div_total = sum(t.amount for t in full.dividends if t.currency == ccy)
    if ccy and {t.currency for t in full.dividends} - {ccy}:
        blockers.append("Dividends exist in a non-account currency; no FX applied.")
    out.append(
        MetricReadiness(
            metric="dividend_income",
            status="unavailable"
            if by_kind["dividends"].state == "never_synced"
            else "available"
            if not blockers
            else "partial",
            value={"dividends": round(div_total, 2)}
            if ccy and by_kind["dividends"].stored_records
            else None,
            basis="Sum of dividend-type payouts as credited (account currency), excluding "
            "interest and capital distributions, which are reported separately.",
            blockers=blockers,
        )
    )

    # --- fees and interest
    blockers = gate("transactions")
    out.append(
        MetricReadiness(
            metric="cash_fees_and_interest",
            status="unavailable"
            if by_kind["transactions"].state == "never_synced"
            else "available"
            if not blockers
            else "partial",
            value={
                "fees": round(sum(t.amount for t in full.cash_fees if t.currency == ccy), 2),
                "interest": round(sum(t.amount for t in full.interest if t.currency == ccy), 2),
            }
            if ccy and by_kind["transactions"].stored_records
            else None,
            basis="FEE, INTEREST_ON_FREE_CASH and LENDING_INTEREST transactions "
            "(account currency). "
            "Per-order fees/taxes are reported separately in order_fees.",
            blockers=blockers,
        )
    )

    # --- realised P/L (broker-reported per fill) reconciled with the account summary
    blockers = gate("orders")
    reported = sum(t.amount for t in full.realised_pnl_reported if t.currency == ccy)
    with engine.connect() as c:
        broker_total = c.execute(
            select(broker_sync_run.c.broker_realised_pnl)
            .where(
                broker_sync_run.c.environment == environment,
                broker_sync_run.c.account_key == account_key,
                broker_sync_run.c.kind == "orders",
                broker_sync_run.c.status == "complete",
            )
            .order_by(broker_sync_run.c.id.desc())
            .limit(1)
        ).first()
    broker_realised = _f(broker_total[0]) if broker_total else None
    if broker_realised is None:
        blockers.append("No broker all-time realised P/L was captured to reconcile against.")
    elif abs(broker_realised - reported) > 0.005 * max(1, full.buy_count + full.sell_count) + 0.01:
        blockers.append(
            "Sum of per-fill realised P/L does not equal the broker's all-time realised P/L "
            f"(difference {reported - broker_realised:+.2f})."
        )
    out.append(
        MetricReadiness(
            metric="realised_pnl_reported",
            status="unavailable"
            if by_kind["orders"].state == "never_synced"
            else "available"
            if not blockers
            else "partial",
            value={"sum_of_fills": round(reported, 2), "broker_all_time": round(broker_realised, 2)}
            if ccy and broker_realised is not None and by_kind["orders"].stored_records
            else None,
            basis="Broker-reported realisedProfitLoss summed over TRADE fills, compared with the "
            "account summary's all-time realised P/L captured at the same sync. Whether the "
            "broker figure includes dividends or fees is not documented.",
            blockers=blockers,
        )
    )

    # --- everything that needs valuations
    for metric, extra in (
        (
            "money_weighted_return_xirr",
            "Needs verified external flows since inception AND a "
            "verified start/terminal valuation in a single currency.",
        ),
        ("time_weighted_return", "Needs portfolio value immediately before every external flow."),
        (
            "portfolio_equity_curve",
            "Needs a daily valuation series (positions x historical "
            "prices x FX); a curve built from current holdings would be speculative.",
        ),
        ("total_return_pct", "Needs the above plus separating deposits from investment profit."),
        (
            "benchmark_comparison",
            "Needs a verified portfolio return series on the same dates "
            "and currency as the benchmark.",
        ),
    ):
        out.append(
            MetricReadiness(
                metric=metric,
                status="unavailable",
                basis="Not calculated in this milestone.",
                blockers=[_CURVE_BLOCKER, extra],
            )
        )
    return out

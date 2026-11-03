"""Deliberate, read-only sync of Trading 212 account history into PostgreSQL.

Properties:
- Idempotent: every row has a stable broker identifier under a unique constraint; a re-delivered
  identical record is counted as a duplicate, never inserted twice. A record whose content
  changed is NOT overwritten; it is counted as conflicting and surfaced.
- Recoverable: progress is committed per page. A run that stops early is recorded as
  `interrupted`/`failed` with a stable error code; re-running restarts from the newest page and
  skips what is stored. A run left `running` by a dead process is marked interrupted.
- Incremental only after a completed full backfill: it stops at the first page whose valid
  records are all already stored. Before a completed backfill every sync is a full one.
- Nothing is inferred: invalid records are rejected with value-free reason codes and counted.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import Table, and_, insert, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from marketops.broker.t212 import BrokerError, HistoryKind, HistoryPacer
from marketops.history.parse import Record, RecordInvalid, parse_record
from marketops.persistence.models import (
    broker_cash_transaction,
    broker_dividend,
    broker_order_fill,
    broker_sync_run,
    init_schema,
)

logger = logging.getLogger(__name__)

KINDS: tuple[HistoryKind, ...] = ("orders", "transactions", "dividends")
TABLES: dict[HistoryKind, Table] = {
    "orders": broker_order_fill,
    "transactions": broker_cash_transaction,
    "dividends": broker_dividend,
}
MAX_PAGES = 2000  # 100k records per kind; hitting it is reported, never silently truncated

_lock = threading.Lock()


class SyncBusy(Exception):
    """Another sync is already running in this process."""


class HistorySource(Protocol):
    @property
    def environment(self) -> str: ...

    def get_account_summary(self) -> dict[str, Any]: ...

    def get_history_page(
        self,
        kind: HistoryKind,
        next_page_path: str | None = None,
        pacer: HistoryPacer | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]: ...


@dataclass
class SyncResult:
    kind: HistoryKind
    run_id: int
    mode: str
    status: str
    pages_fetched: int = 0
    seen: int = 0
    inserted: int = 0
    duplicate: int = 0
    rejected: int = 0
    conflicting: int = 0
    reached_end: bool = False
    error_code: str | None = None
    reject_reasons: dict[str, int] = field(default_factory=dict)


def sync_in_progress() -> bool:
    return _lock.locked()


def make_account_key(environment: str, account_id: object) -> str:
    """Opaque stable key; the raw account number is never persisted."""
    return hashlib.sha256(f"{environment}:{account_id}".encode()).hexdigest()[:16]


def _now() -> datetime:
    return datetime.now(UTC)


def has_completed_backfill(engine: Engine, environment: str, account_key: str, kind: str) -> bool:
    with engine.connect() as c:
        return (
            c.execute(
                select(broker_sync_run.c.id)
                .where(
                    broker_sync_run.c.environment == environment,
                    broker_sync_run.c.account_key == account_key,
                    broker_sync_run.c.kind == kind,
                    broker_sync_run.c.mode == "full",
                    broker_sync_run.c.status == "complete",
                    broker_sync_run.c.reached_end.is_(True),
                )
                .limit(1)
            ).first()
            is not None
        )


def recover_stale_runs(engine: Engine, environment: str, account_key: str) -> int:
    """Runs still 'running' while we hold the lock were left by a dead process."""
    with engine.begin() as c:
        res = c.execute(
            update(broker_sync_run)
            .where(
                broker_sync_run.c.environment == environment,
                broker_sync_run.c.account_key == account_key,
                broker_sync_run.c.status == "running",
            )
            .values(status="interrupted", error_code="process_terminated", finished_utc=_now())
        )
        return res.rowcount or 0


def _store_page(
    engine: Engine,
    table: Table,
    environment: str,
    account_key: str,
    run_id: int,
    records: list[Record],
    result: SyncResult,
) -> int:
    """Insert new records, count duplicates/conflicts; returns the number newly inserted."""
    unique: dict[str, Record] = {}
    for rec in records:
        prior = unique.get(rec.source_id)
        if prior is None:
            unique[rec.source_id] = rec
        elif prior.content_hash == rec.content_hash:
            result.duplicate += 1
        else:
            result.conflicting += 1
    if not unique:
        return 0
    with engine.begin() as c:
        stored: dict[str, str] = dict(
            c.execute(
                select(table.c.source_id, table.c.content_hash).where(
                    and_(
                        table.c.environment == environment,
                        table.c.account_key == account_key,
                        table.c.source_id.in_(list(unique)),
                    )
                )
            ).all()
        )
        fresh: list[dict[str, Any]] = []
        for sid, rec in unique.items():
            if sid not in stored:
                fresh.append(
                    {
                        **rec.row,
                        "environment": environment,
                        "account_key": account_key,
                        "content_hash": rec.content_hash,
                        "first_seen_utc": _now(),
                        "sync_run_id": run_id,
                    }
                )
            elif stored[sid] == rec.content_hash:
                result.duplicate += 1
            else:
                result.conflicting += 1
        if fresh:
            c.execute(insert(table), fresh)
    result.inserted += len(fresh)
    return len(fresh)


def _progress(engine: Engine, result: SyncResult, **extra: Any) -> None:
    with engine.begin() as c:
        c.execute(
            update(broker_sync_run)
            .where(broker_sync_run.c.id == result.run_id)
            .values(
                pages_fetched=result.pages_fetched,
                records_seen=result.seen,
                records_inserted=result.inserted,
                records_duplicate=result.duplicate,
                records_rejected=result.rejected,
                records_conflicting=result.conflicting,
                reject_reasons=result.reject_reasons,
                reached_end=result.reached_end,
                error_code=result.error_code,
                **extra,
            )
        )


def sync_kind(
    engine: Engine,
    client: HistorySource,
    kind: HistoryKind,
    environment: str,
    account_key: str,
    account_currency: str | None,
    *,
    full: bool,
    pacer: HistoryPacer,
    broker_realised_pnl: Any = None,
) -> SyncResult:
    incremental = not full and has_completed_backfill(engine, environment, account_key, kind)
    mode = "incremental" if incremental else "full"
    with engine.begin() as c:
        pk: Any = c.execute(
            insert(broker_sync_run).values(
                environment=environment,
                account_key=account_key,
                kind=kind,
                mode=mode,
                status="running",
                started_utc=_now(),
                reached_end=False,
                pages_fetched=0,
                records_seen=0,
                records_inserted=0,
                records_duplicate=0,
                records_rejected=0,
                records_conflicting=0,
                reject_reasons={},
                account_currency=account_currency,
                broker_realised_pnl=broker_realised_pnl if kind == "orders" else None,
            )
        ).inserted_primary_key
    run_id = int(pk[0])
    result = SyncResult(kind=kind, run_id=run_id, mode=mode, status="running")
    reasons: Counter[str] = Counter()
    seen_paths: set[str] = set()
    next_path: str | None = None
    status = "complete"
    try:
        while True:
            if result.pages_fetched >= MAX_PAGES:
                status, result.error_code = "interrupted", "page_limit"
                break
            items, nxt = client.get_history_page(kind, next_path, pacer)
            result.pages_fetched += 1
            if not items and nxt:
                status, result.error_code = "interrupted", "empty_page_with_cursor"
                break
            records: list[Record] = []
            for raw in items:
                result.seen += 1
                try:
                    records.append(parse_record(kind, raw, account_currency))
                except RecordInvalid as exc:
                    result.rejected += 1
                    reasons[exc.reason] += 1
            result.reject_reasons = dict(reasons)
            new = _store_page(
                engine, TABLES[kind], environment, account_key, run_id, records, result
            )
            _progress(engine, result)
            if nxt is None:
                result.reached_end = True
                break
            if incremental and records and new == 0:
                break  # reached data we already hold; older pages are already stored
            if nxt in seen_paths or nxt == next_path:
                status, result.error_code = "interrupted", "pagination_loop"
                break
            seen_paths.add(nxt)
            next_path = nxt
    except BrokerError as exc:
        status = "interrupted" if result.pages_fetched else "failed"
        result.error_code = exc.code
    except SQLAlchemyError as exc:
        logger.warning("history sync storage failure: %s", type(exc).__name__)
        status, result.error_code = "interrupted", "storage"
    except Exception as exc:  # defensive: a bug must leave an auditable, recoverable record
        logger.error("history sync unexpected failure: %s", type(exc).__name__)
        status, result.error_code = "interrupted", "internal"
    result.status = status
    try:
        _progress(engine, result, status=status, finished_utc=_now())
    except SQLAlchemyError as exc:
        logger.warning("history sync final update failed: %s", type(exc).__name__)
    logger.info(
        "history_sync kind=%s mode=%s status=%s pages=%d seen=%d inserted=%d dup=%d "
        "rejected=%d conflicts=%d code=%s",
        kind,
        mode,
        status,
        result.pages_fetched,
        result.seen,
        result.inserted,
        result.duplicate,
        result.rejected,
        result.conflicting,
        result.error_code,
    )
    return result


def account_identity(client: HistorySource) -> tuple[str, str, Any]:
    """(account_key, currency, broker realised P/L) from the account summary; fails closed."""
    summary = client.get_account_summary()
    account_id = summary.get("id")
    currency = summary.get("currency")
    if isinstance(account_id, bool) or not isinstance(account_id, int | str) or not account_id:
        raise BrokerError("Trading 212 account summary has no usable account id.", "schema")
    if not isinstance(currency, str) or not currency:
        raise BrokerError("Trading 212 account summary has no account currency.", "schema")
    investments = summary.get("investments")
    realised = investments.get("realizedProfitLoss") if isinstance(investments, dict) else None
    if isinstance(realised, bool) or not isinstance(realised, int | float):
        realised = None
    return make_account_key(client.environment, account_id), currency, realised


def sync_account_history(
    engine: Engine,
    client: HistorySource,
    *,
    full: bool = False,
    kinds: tuple[HistoryKind, ...] = KINDS,
    pacer: HistoryPacer | None = None,
    identity: tuple[str, str, Any] | None = None,
) -> list[SyncResult]:
    if not _lock.acquire(blocking=False):
        raise SyncBusy
    try:
        init_schema(engine)
        key, currency, realised = identity or account_identity(client)
        recover_stale_runs(engine, client.environment, key)
        pacer = pacer or HistoryPacer()
        results: list[SyncResult] = []
        for kind in kinds:
            res = sync_kind(
                engine,
                client,
                kind,
                client.environment,
                key,
                currency,
                full=full,
                pacer=pacer,
                broker_realised_pnl=realised,
            )
            results.append(res)
            if res.error_code == "auth":  # credentials are wrong for every endpoint
                break
        return results
    finally:
        _lock.release()

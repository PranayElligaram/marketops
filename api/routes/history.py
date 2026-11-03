"""Private Trading 212 account-history routes (read-only import, then read from PostgreSQL).

Every route requires the local access token. Listing/summary routes read only PostgreSQL and
never contact the broker; the single broker-touching route is the deliberate POST /history/sync.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from api.security import private_access
from marketops.broker.t212 import (
    BrokerError,
    HistoryKind,
    Trading212ReadOnlyClient,
    require_environment_allowed,
)
from marketops.config import get_settings
from marketops.history import read
from marketops.history.sync import (
    KINDS,
    SyncBusy,
    account_identity,
    sync_account_history,
    sync_in_progress,
)
from marketops.persistence import database
from marketops.persistence.models import init_schema

logger = logging.getLogger("marketops.history")
router = APIRouter(prefix="/history", tags=["history"], dependencies=[Depends(private_access)])


class SyncRequest(BaseModel):
    full: bool = False  # force a complete re-pass (idempotent); default is automatic


class SyncStarted(BaseModel):
    started: bool
    environment: str
    account_currency: str
    note: str


class Page(BaseModel):
    total: int
    items: list[dict[str, Any]]


def _engine() -> Engine:
    try:
        engine = database.get_engine()
        init_schema(engine)  # idempotent create-if-missing (project convention, no Alembic)
        return engine
    except (RuntimeError, SQLAlchemyError):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "PostgreSQL is not available for account history."
        ) from None


def _range(
    date_from: Annotated[date | None, Query()] = None,
    date_to: Annotated[date | None, Query()] = None,
) -> tuple[date | None, date | None]:
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "date_from is after date_to.")
    return date_from, date_to


Window = Annotated[tuple[date | None, date | None], Depends(_range)]
Limit = Annotated[int, Query(ge=1, le=500)]
Offset = Annotated[int, Query(ge=0)]


def _scope() -> tuple[Engine, str, str | None]:
    engine = _engine()
    env = get_settings().t212_environment
    return engine, env, read.latest_account_key(engine, env)


def _run_sync(client: Trading212ReadOnlyClient, engine: Engine, full: bool, identity: Any) -> None:
    try:
        sync_account_history(engine, client, full=full, identity=identity)
    except SyncBusy:
        logger.info("history sync skipped: another sync is running")
    except Exception as exc:  # background task: never propagate; runs are audited in the DB
        logger.error("history sync failed: %s", type(exc).__name__)


@router.post("/sync", response_model=SyncStarted, status_code=status.HTTP_202_ACCEPTED)
def start_sync(body: SyncRequest, background: BackgroundTasks) -> SyncStarted:
    """Deliberate action: downloads account history (read-only GETs) in the background."""
    settings = get_settings()
    if not settings.broker_configured:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Trading 212 read-only keys are not configured."
        )
    engine = _engine()
    if sync_in_progress():
        raise HTTPException(status.HTTP_409_CONFLICT, "A history sync is already running.")
    try:
        require_environment_allowed(settings.t212_environment, settings.t212_allow_live)
        client = Trading212ReadOnlyClient(
            settings.t212_api_key, settings.t212_api_secret, settings.t212_environment
        )
        identity = account_identity(client)
    except BrokerError as exc:
        code = {
            "live_disabled": status.HTTP_403_FORBIDDEN,
            "rate_limit": status.HTTP_429_TOO_MANY_REQUESTS,
        }.get(exc.code, status.HTTP_502_BAD_GATEWAY)
        raise HTTPException(code, str(exc)) from None
    background.add_task(_run_sync, client, engine, body.full, identity)
    return SyncStarted(
        started=True,
        environment=settings.t212_environment,
        account_currency=identity[1],
        note="Sync started. Progress is shown by GET /history/status; it is rate limited "
        "(20 requests/minute per endpoint) so a large history takes several minutes.",
    )


@router.get("/status", response_model=read.Coverage)
def sync_status() -> read.Coverage:
    engine, env, key = _scope()
    if key is None:
        return read.Coverage(
            environment=env,
            account_currency=None,
            kinds=[read.KindCoverage(kind=k, state="never_synced") for k in KINDS],
            sync_running=sync_in_progress(),
        )
    return read.coverage(engine, env, key, sync_in_progress())


@router.get("/summary", response_model=read.ActivitySummary)
def summary(window: Window) -> read.ActivitySummary:
    engine, env, key = _scope()
    if key is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No account history has been synced yet.")
    cov = read.coverage(engine, env, key, sync_in_progress())
    return read.summarise_activity(engine, env, key, *window, account_currency=cov.account_currency)


@router.get("/readiness", response_model=list[read.MetricReadiness])
def readiness() -> list[read.MetricReadiness]:
    engine, env, key = _scope()
    if key is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No account history has been synced yet.")
    cov = read.coverage(engine, env, key, sync_in_progress())
    return read.performance_readiness(engine, env, key, cov)


def _listing(
    kind: HistoryKind, window: tuple[date | None, date | None], limit: int, offset: int
) -> Page:
    engine, env, key = _scope()
    if key is None:
        return Page(total=0, items=[])
    rows, total = read.list_rows(engine, kind, env, key, *window, limit=limit, offset=offset)
    return Page(total=total, items=rows)


@router.get("/transactions", response_model=Page)
def transactions(window: Window, limit: Limit = 200, offset: Offset = 0) -> Page:
    return _listing("transactions", window, limit, offset)


@router.get("/orders", response_model=Page)
def orders(window: Window, limit: Limit = 200, offset: Offset = 0) -> Page:
    return _listing("orders", window, limit, offset)


@router.get("/dividends", response_model=Page)
def dividends(window: Window, limit: Limit = 200, offset: Offset = 0) -> Page:
    return _listing("dividends", window, limit, offset)

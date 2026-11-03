"""Private historical-performance route: stored history reconciled against one live snapshot.

Contacts the broker only through the existing cached, read-only positions/summary snapshot.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from api.routes import portfolio
from api.security import private_access
from marketops.config import get_settings
from marketops.history import read
from marketops.history.sync import sync_in_progress
from marketops.performance.report import HistoricalPerformance, build_report
from marketops.persistence import database
from marketops.persistence.models import init_schema
from marketops.portfolio.analytics import PortfolioOverview

router = APIRouter(
    prefix="/performance", tags=["performance"], dependencies=[Depends(private_access)]
)


def _engine() -> Engine:
    try:
        engine = database.get_engine()
        init_schema(engine)
        return engine
    except (RuntimeError, SQLAlchemyError):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "PostgreSQL is not available for account history."
        ) from None


def _live_snapshot() -> tuple[PortfolioOverview | None, str | None]:
    try:
        return portfolio.overview(), None
    except HTTPException as exc:
        return None, f"Live snapshot unavailable ({exc.status_code}): {exc.detail}"


@router.get("/historical", response_model=HistoricalPerformance)
def historical() -> HistoricalPerformance:
    engine = _engine()
    env = get_settings().t212_environment
    key = read.latest_account_key(engine, env)
    if key is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No account history has been synced yet.")
    cov = read.coverage(engine, env, key, sync_in_progress())
    snapshot, problem = _live_snapshot()
    if snapshot is not None and snapshot.broker_environment not in (None, env):
        snapshot, problem = None, "Live snapshot belongs to a different environment."
    report = build_report(engine, env, key, cov, snapshot)
    if problem:
        report.notes.append(problem)
    return report

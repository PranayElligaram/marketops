"""Database connectivity. SQL models and migrations arrive in Phase 3."""

import logging
from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from marketops.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    database_url = get_settings().database_url
    if not database_url:
        raise RuntimeError("MARKETOPS_DATABASE_URL is not configured")
    return create_engine(
        database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 3},
    )


def database_is_ready() -> bool:
    """Confirm a genuine DB round trip, rather than reporting a configured URL as healthy."""
    if not get_settings().database_url:
        return False
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
    except (SQLAlchemyError, OSError, ImportError) as exc:
        logger.warning("PostgreSQL readiness check failed: %s", type(exc).__name__)
        return False
    return True

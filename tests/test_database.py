"""Database readiness tests without external services."""

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from marketops.config import Settings
from marketops.persistence import database


def test_unconfigured_database_is_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None, database_url=None)
    monkeypatch.setattr(database, "get_settings", lambda: settings)
    assert database.database_is_ready() is False


def test_readiness_executes_database_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    engine: Engine = create_engine("sqlite+pysqlite:///:memory:")
    settings = Settings(_env_file=None, database_url="set")
    monkeypatch.setattr(database, "get_settings", lambda: settings)
    monkeypatch.setattr(database, "get_engine", lambda: engine)
    try:
        assert database.database_is_ready() is True
    finally:
        engine.dispose()


def test_database_connection_failure_is_not_reported_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # SQLite must not create the missing parent directory automatically.
    engine: Engine = create_engine(f"sqlite:///{tmp_path / 'does-not-exist' / 'db.sqlite'}")
    settings = Settings(_env_file=None, database_url="set")
    monkeypatch.setattr(database, "get_settings", lambda: settings)
    monkeypatch.setattr(database, "get_engine", lambda: engine)
    try:
        assert database.database_is_ready() is False
    finally:
        engine.dispose()

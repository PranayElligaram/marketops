"""Hermetic test settings: the developer's real .env and MARKETOPS_* variables never leak in."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from marketops.config import Settings, get_settings
from marketops.history import sync
from marketops.persistence import database
from marketops.persistence.models import init_schema
from tests.history_fixtures import KEY, SECRET, TOKEN, Env, FakeSource, sqlite_engine

# Needed only by the opt-in Postgres tests, which pass them explicitly via the environment.
_KEEP = {"MARKETOPS_DATABASE_URL", "MARKETOPS_RUN_INTEGRATION"}


@pytest.fixture(autouse=True)
def _hermetic_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in [n for n in os.environ if n.startswith("MARKETOPS_") and n not in _KEEP]:
        monkeypatch.delenv(name)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


_REAL_PACER = sync.HistoryPacer


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Env]:
    """API + scripted broker + SQLite: demo environment, token and keys configured."""
    engine = sqlite_engine(tmp_path)
    init_schema(engine)
    for k, v in {
        "MARKETOPS_AGENT_API_TOKEN": TOKEN,
        "MARKETOPS_T212_API_KEY": KEY,
        "MARKETOPS_T212_API_SECRET": SECRET,
        "MARKETOPS_T212_ENVIRONMENT": "demo",
    }.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    monkeypatch.setattr(database, "get_engine", lambda: engine)
    e = Env(engine, TestClient(app))

    def factory(api_key: str, api_secret: str, environment: str) -> FakeSource:
        e.constructed += 1
        e.source.environment = environment
        return e.source

    monkeypatch.setattr("api.routes.history.Trading212ReadOnlyClient", factory)
    monkeypatch.setattr(sync, "HistoryPacer", lambda: _REAL_PACER(0, sleep=lambda s: None))
    yield e
    engine.dispose()

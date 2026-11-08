"""Opt-in: the research journal API against a real PostgreSQL server (throwaway database).

All rows are synthetic. The journal has no broker dependency; one test proves that.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine
from sqlalchemy.engine.url import make_url

import api.routes.journal as journal_module
from api.main import app
from marketops.config import get_settings
from marketops.persistence import database
from marketops.persistence.models import init_schema

pytestmark = pytest.mark.skipif(
    os.getenv("MARKETOPS_RUN_INTEGRATION") != "1" or not os.getenv("MARKETOPS_DATABASE_URL"),
    reason="Set MARKETOPS_RUN_INTEGRATION=1 with MARKETOPS_DATABASE_URL to run on PostgreSQL",
)

TOKEN = "journal-pg-test-token-0123456789"
HDR = {"X-MarketOps-Token": TOKEN}
TODAY = datetime.now(UTC).date()


class Pg:
    def __init__(self, base: URL, name: str) -> None:
        self.base, self.name = base, name
        self.engine = self.connect()

    def connect(self) -> Engine:
        return create_engine(self.base.set(database=self.name))

    def restart(self) -> None:
        """Drop every pooled connection and reconnect, as an API restart would."""
        self.engine.dispose()
        self.engine = self.connect()


@pytest.fixture
def pg(monkeypatch: pytest.MonkeyPatch) -> Iterator[Pg]:
    base = make_url(os.environ["MARKETOPS_DATABASE_URL"])
    name = f"marketops_journal_{uuid.uuid4().hex[:10]}"
    admin = create_engine(base, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f'CREATE DATABASE "{name}"'))
    handle = Pg(base, name)
    init_schema(handle.engine)
    monkeypatch.setenv("MARKETOPS_AGENT_API_TOKEN", TOKEN)
    get_settings.cache_clear()
    monkeypatch.setattr(database, "get_engine", lambda: handle.engine)
    try:
        yield handle
    finally:
        handle.engine.dispose()
        with admin.connect() as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()
        get_settings.cache_clear()


def _idea(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "symbol": "msft",
        "decision_day": TODAY.isoformat(),
        "thesis": "Synthetic breakout hypothesis for testing",
        "risk_note": "Synthetic risk note for testing",
        "hypothetical_entry_gbp": "123.45678901",
        "hypothetical_units": "3.12345678",
        "planned_stop_gbp": "100.5",
    }
    return base | over


def _close(exit_: str = "130.00") -> dict[str, object]:
    return {
        "outcome_day": TODAY.isoformat(),
        "hypothetical_exit_gbp": exit_,
        "outcome_note": "Synthetic outcome for testing",
    }


def test_create_read_and_decimal_date_persistence(pg: Pg) -> None:
    client = TestClient(app)
    old = (TODAY - timedelta(days=400)).isoformat()
    made = client.post("/journal/ideas", json=_idea(decision_day=old), headers=HDR)
    assert made.status_code == 201, made.text
    body = made.json()
    assert body["retrospective"] is True and body["symbol"] == "MSFT"
    listed = client.get("/journal/ideas", headers=HDR).json()
    assert [r["id"] for r in listed] == [body["id"]]
    row = listed[0]
    # Precision is bounded by NUMERIC(24, 8); values never pass through binary floating point.
    assert Decimal(row["hypothetical_entry_gbp"]) == Decimal("123.45678901").quantize(
        Decimal("1e-8")
    )
    assert Decimal(row["hypothetical_units"]) == Decimal("3.12345678")
    assert Decimal(row["planned_stop_gbp"]) == Decimal("100.5")
    assert row["decision_day"] == old


def test_prospective_vs_retrospective_flag_persists(pg: Pg) -> None:
    client = TestClient(app)
    yesterday = (TODAY - timedelta(days=1)).isoformat()
    a = client.post("/journal/ideas", json=_idea(), headers=HDR).json()
    b = client.post("/journal/ideas", json=_idea(decision_day=yesterday), headers=HDR).json()
    assert (a["retrospective"], b["retrospective"]) == (False, True)
    with pg.engine.connect() as con:
        flags = dict(con.execute(text("select id, retrospective from research_journal")).all())  # type: ignore[arg-type]
    assert flags == {a["id"]: False, b["id"]: True}


def test_close_and_duplicate_closure_rejected(pg: Pg) -> None:
    client = TestClient(app)
    idea = client.post("/journal/ideas", json=_idea(), headers=HDR).json()
    url = f"/journal/ideas/{idea['id']}/close"
    first = client.post(url, json=_close(), headers=HDR)
    assert first.status_code == 200, first.text
    expected = (Decimal("130.00") - Decimal("123.45678901")) * Decimal("3.12345678")
    assert Decimal(first.json()["hypothetical_gross_pnl_gbp"]) == expected
    second = client.post(url, json=_close(exit_="1.00"), headers=HDR)
    assert second.status_code == 409
    again = client.get("/journal/ideas", headers=HDR).json()[0]
    assert Decimal(again["hypothetical_exit_gbp"]) == Decimal("130.00")


def test_concurrent_closure_only_one_wins(pg: Pg) -> None:
    idea = TestClient(app).post("/journal/ideas", json=_idea(), headers=HDR).json()
    url = f"/journal/ideas/{idea['id']}/close"

    def attempt(price: str) -> int:
        return TestClient(app).post(url, json=_close(exit_=price), headers=HDR).status_code

    with ThreadPoolExecutor(max_workers=4) as pool:
        codes = sorted(pool.map(attempt, ["110", "111", "112", "113"]))
    assert codes == [200, 409, 409, 409]


def test_persistence_across_restart(pg: Pg) -> None:
    client = TestClient(app)
    idea = client.post("/journal/ideas", json=_idea(), headers=HDR).json()
    client.post(f"/journal/ideas/{idea['id']}/close", json=_close(), headers=HDR)
    pg.restart()
    rows = TestClient(app).get("/journal/ideas", headers=HDR).json()
    assert len(rows) == 1 and rows[0]["status"] == "closed"
    assert Decimal(rows[0]["hypothetical_exit_gbp"]) == Decimal("130.00")


def test_authentication_enforced_and_nothing_written_without_it(pg: Pg) -> None:
    client = TestClient(app)
    assert client.get("/journal/ideas").status_code == 401
    assert client.post("/journal/ideas", json=_idea()).status_code == 401
    bad = {"X-MarketOps-Token": "wrong"}
    assert client.post("/journal/ideas", json=_idea(), headers=bad).status_code == 401
    assert client.post("/journal/ideas/1/close", json=_close()).status_code == 401
    with pg.engine.connect() as con:
        assert con.execute(text("select count(*) from research_journal")).scalar_one() == 0


def test_journal_has_no_broker_or_order_surface(pg: Pg) -> None:
    paths = list(TestClient(app).app.openapi()["paths"])  # type: ignore[attr-defined]
    journal_paths = [p for p in paths if p.startswith("/journal")]
    assert journal_paths
    assert not any(w in p for p in journal_paths for w in ("order", "trade", "execute", "fill"))
    source = Path(journal_module.__file__).read_text(encoding="utf-8").lower()
    assert "trading212" not in source and "t212" not in source


def test_init_schema_adds_closed_utc_to_a_legacy_table(pg: Pg) -> None:
    with pg.engine.begin() as con:
        con.execute(text("ALTER TABLE research_journal DROP COLUMN closed_utc"))
    init_schema(pg.engine)
    init_schema(pg.engine)  # idempotent
    client = TestClient(app)
    idea = client.post("/journal/ideas", json=_idea(), headers=HDR).json()
    closed = client.post(f"/journal/ideas/{idea['id']}/close", json=_close(), headers=HDR).json()
    assert closed["outcome_backfilled"] is False and closed["closed_utc"]
    summary = client.get("/journal/summary", headers=HDR).json()
    assert summary["scored_contemporaneous"] == 1

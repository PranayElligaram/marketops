"""Opt-in: the history schema and sync against a real PostgreSQL server.

Runs in a throwaway database created and dropped on the configured server, so synthetic rows
never touch the application database.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, func, insert, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import IntegrityError

from marketops.history import read
from marketops.history.sync import make_account_key, sync_account_history
from marketops.persistence.models import broker_cash_transaction, init_schema
from tests.history_fixtures import FakeSource, div, fast_pacer, standard_pages, tx

pytestmark = pytest.mark.skipif(
    os.getenv("MARKETOPS_RUN_INTEGRATION") != "1" or not os.getenv("MARKETOPS_DATABASE_URL"),
    reason="Set MARKETOPS_RUN_INTEGRATION=1 with MARKETOPS_DATABASE_URL to run on PostgreSQL",
)


@pytest.fixture
def pg() -> Iterator[Engine]:
    base = make_url(os.environ["MARKETOPS_DATABASE_URL"])
    name = f"marketops_test_{uuid.uuid4().hex[:10]}"
    admin = create_engine(base, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f'CREATE DATABASE "{name}"'))
    engine = create_engine(base.set(database=name))
    try:
        init_schema(engine)
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def test_schema_creation_is_idempotent(pg: Engine) -> None:
    init_schema(pg)
    init_schema(pg)
    with pg.connect() as c:
        tables = {
            r[0]
            for r in c.execute(
                text("select table_name from information_schema.tables where table_schema='public'")
            )
        }
    assert {
        "broker_sync_run",
        "broker_order_fill",
        "broker_cash_transaction",
        "broker_dividend",
    } <= tables


def test_full_sync_incremental_and_rerun_on_postgres(pg: Engine) -> None:
    src = FakeSource(standard_pages())
    first = sync_account_history(pg, src, pacer=fast_pacer())
    assert [r.status for r in first] == ["complete"] * 3
    again = sync_account_history(pg, src, pacer=fast_pacer(), full=True)
    assert [r.inserted for r in again] == [0, 0, 0]
    inc = sync_account_history(pg, src, pacer=fast_pacer())
    assert {r.mode for r in inc} == {"incremental"} and [r.inserted for r in inc] == [0, 0, 0]
    key = make_account_key("demo", 424242)
    cov = read.coverage(pg, "demo", key, False)
    assert {k.kind: (k.state, k.stored_records) for k in cov.kinds} == {
        "orders": ("complete", 4),
        "transactions": ("complete", 4),
        "dividends": ("complete", 2),
    }
    summary = read.summarise_activity(pg, "demo", key, account_currency="GBP")
    assert summary.deposits[0].amount == 700.0 and summary.dividends[0].amount == 7.2


def test_database_enforces_uniqueness_per_environment_and_account(pg: Engine) -> None:
    sync_account_history(pg, FakeSource(standard_pages()), pacer=fast_pacer())
    key = make_account_key("demo", 424242)
    row = {
        "environment": "demo",
        "account_key": key,
        "source_id": "T1",
        "tx_type": "DEPOSIT",
        "category": "deposit",
        "occurred_at": "2024-02-03T12:00:00+00:00",
        "amount": Decimal("1"),
        "currency": "GBP",
        "content_hash": "x",
        "first_seen_utc": "2024-02-03T12:00:00+00:00",
        "sync_run_id": 1,
    }
    with pytest.raises(IntegrityError), pg.begin() as c:
        c.execute(insert(broker_cash_transaction).values(**row))
    # the same broker reference under another environment or account is a different record
    for other in ({"environment": "live"}, {"account_key": "0" * 16}):
        with pg.begin() as c:
            c.execute(insert(broker_cash_transaction).values(**{**row, **other}))


def test_decimal_precision_and_timezones_round_trip(pg: Engine) -> None:
    pages = standard_pages()
    pages["transactions"] = [[tx("P1", "DEPOSIT", 0.12345678)]]
    pages["dividends"] = [[div("P2", 1234567.89)]]
    sync_account_history(pg, FakeSource(pages), pacer=fast_pacer())
    with pg.connect() as c:
        amount, when = c.execute(
            select(broker_cash_transaction.c.amount, broker_cash_transaction.c.occurred_at)
        ).one()
    assert amount == Decimal("0.12345678")
    assert when.utcoffset().total_seconds() == 0 and when.isoformat().startswith("2024-02-05T12:00")


def test_interruption_and_recovery_on_postgres(pg: Engine) -> None:
    from marketops.broker.t212 import BrokerError

    src = FakeSource(standard_pages())
    src.fail_at[("transactions", 1)] = BrokerError("down", "unavailable")
    res = {r.kind: r for r in sync_account_history(pg, src, pacer=fast_pacer())}
    assert res["transactions"].status == "interrupted" and res["transactions"].inserted == 2
    src.fail_at.clear()
    res = {r.kind: r for r in sync_account_history(pg, src, pacer=fast_pacer())}
    assert res["transactions"].status == "complete" and res["transactions"].mode == "full"
    with pg.connect() as c:
        n = c.execute(select(func.count()).select_from(broker_cash_transaction)).scalar_one()
    assert n == 4


def test_historical_performance_report_on_postgres(pg: Engine) -> None:
    from marketops.performance.report import build_report
    from tests.test_perf_report import snapshot

    sync_account_history(pg, FakeSource(standard_pages()), pacer=fast_pacer())
    key = make_account_key("demo", 424242)
    cov = read.coverage(pg, "demo", key, False)
    rep = build_report(pg, "demo", key, cov, snapshot())
    assert rep.overall_status == "verified" and rep.cash_ledger.status == "match"
    gain = next(m for m in rep.metrics if m.name == "net_gain_on_invested_capital")
    assert gain.value is not None and abs(gain.value["gain"] - 56.2) < 1e-9
    assert rep.xirr is not None and rep.xirr.status == "ok"

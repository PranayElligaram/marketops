"""Sync engine behaviour on SQLite with a scripted in-memory broker (no network)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.engine import Engine

from marketops.broker.t212 import BrokerError
from marketops.history import read, sync
from marketops.history.sync import (
    TABLES,
    SyncBusy,
    make_account_key,
    sync_account_history,
)
from marketops.persistence.models import broker_sync_run
from tests.history_fixtures import (
    FakeSource,
    div,
    fast_pacer,
    order,
    sqlite_engine,
    standard_pages,
    tx,
)

pytestmark = pytest.mark.filterwarnings("ignore:Dialect sqlite.*Decimal")


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    return sqlite_engine(tmp_path)


def _count(engine: Engine, kind: str) -> int:
    with engine.connect() as c:
        return int(c.execute(select(func.count()).select_from(TABLES[kind])).scalar_one())  # type: ignore[index]


def _run(engine: Engine, src: FakeSource, **kw: object) -> dict[str, sync.SyncResult]:
    results = sync_account_history(engine, src, pacer=fast_pacer(), **kw)  # type: ignore[arg-type]
    return {r.kind: r for r in results}


def test_full_sync_follows_cursor_across_pages(engine: Engine) -> None:
    src = FakeSource(standard_pages())
    res = _run(engine, src)
    assert [(k, i) for k, i in src.calls if k == "orders"] == [
        ("orders", 0),
        ("orders", 1),
        ("orders", 2),
    ]
    assert res["orders"].pages_fetched == 3 and res["orders"].inserted == 4
    assert res["transactions"].inserted == 4 and res["dividends"].inserted == 2
    assert all(r.status == "complete" and r.reached_end and r.mode == "full" for r in res.values())
    assert (_count(engine, "orders"), _count(engine, "transactions")) == (4, 4)
    key = make_account_key("demo", 424242)
    cov = read.coverage(engine, "demo", key, False)
    assert {k.kind: k.state for k in cov.kinds} == {
        "orders": "complete",
        "transactions": "complete",
        "dividends": "complete",
    }
    assert cov.account_currency == "GBP"


def test_account_number_is_not_stored(engine: Engine) -> None:
    _run(engine, FakeSource(standard_pages()))
    with engine.connect() as c:
        keys = {r[0] for r in c.execute(select(broker_sync_run.c.account_key))}
    assert keys == {make_account_key("demo", 424242)} and "424242" not in "".join(keys)


def test_rerunning_a_full_sync_is_idempotent(engine: Engine) -> None:
    src = FakeSource(standard_pages())
    _run(engine, src)
    second = _run(engine, src, full=True)
    assert all(r.inserted == 0 and r.status == "complete" for r in second.values())
    assert second["orders"].duplicate == 4 and second["transactions"].duplicate == 4
    assert (
        _count(engine, "orders"),
        _count(engine, "transactions"),
        _count(engine, "dividends"),
    ) == (4, 4, 2)


def test_incremental_fetches_new_records_and_stops_at_known_data(engine: Engine) -> None:
    pages = standard_pages()
    _run(engine, FakeSource(pages))
    newer = copy.deepcopy(pages)
    newer["transactions"][0] = [tx("T6", "DEPOSIT", 10.0, day=25), tx("T5", "FEE", -1, day=24)]
    newer["transactions"] = [
        newer["transactions"][0],
        pages["transactions"][0],
        pages["transactions"][1],
    ]
    src = FakeSource(newer)
    res = _run(engine, src)
    r = res["transactions"]
    assert r.mode == "incremental" and r.status == "complete"
    assert r.inserted == 2 and r.pages_fetched == 2 and not r.reached_end
    assert [c for c in src.calls if c[0] == "transactions"] == [
        ("transactions", 0),
        ("transactions", 1),
    ]
    assert _count(engine, "transactions") == 6
    assert res["orders"].inserted == 0 and res["orders"].pages_fetched == 1  # nothing new: one page


def test_interrupted_sync_is_detectable_and_recoverable(engine: Engine) -> None:
    src = FakeSource(standard_pages())
    src.fail_at[("orders", 2)] = BrokerError("rate limited", "rate_limit")
    first = _run(engine, src, kinds=("orders",))["orders"]
    assert first.status == "interrupted" and first.error_code == "rate_limit"
    assert first.pages_fetched == 2 and first.inserted == 3 and not first.reached_end
    key = make_account_key("demo", 424242)
    cov = {k.kind: k for k in read.coverage(engine, "demo", key, False).kinds}["orders"]
    assert cov.state == "partial" and cov.last_error_code == "rate_limit"
    assert cov.backfill_complete_utc is None and any("incomplete" in i for i in cov.issues)

    src.fail_at.clear()
    again = _run(engine, src, kinds=("orders",))["orders"]
    assert again.mode == "full"  # an incremental run is never trusted before a finished backfill
    assert again.status == "complete" and again.inserted == 1 and again.duplicate == 3
    assert _count(engine, "orders") == 4
    cov = {k.kind: k for k in read.coverage(engine, "demo", key, False).kinds}["orders"]
    assert cov.state == "complete"


def test_failure_on_first_page_is_failed_not_complete(engine: Engine) -> None:
    src = FakeSource(standard_pages())
    src.fail_at[("dividends", 0)] = BrokerError("scope", "scope")
    res = _run(engine, src)
    assert res["dividends"].status == "failed" and res["dividends"].error_code == "scope"
    assert (
        res["orders"].status == "complete"
    )  # a missing scope on one endpoint does not stop others
    cov = {
        k.kind: k.state
        for k in read.coverage(engine, "demo", make_account_key("demo", 424242), False).kinds
    }
    assert cov["dividends"] == "partial"


def test_auth_failure_stops_remaining_endpoints(engine: Engine) -> None:
    src = FakeSource(standard_pages())
    src.fail_at[("orders", 0)] = BrokerError("bad key", "auth")
    results = sync_account_history(engine, src, pacer=fast_pacer())
    assert [r.kind for r in results] == ["orders"] and results[0].status == "failed"


def test_stale_incremental_failure_marks_data_stale_not_lost(engine: Engine) -> None:
    src = FakeSource(standard_pages())
    _run(engine, src)
    src.fail_at[("transactions", 0)] = BrokerError("down", "unavailable")
    res = _run(engine, src)
    assert res["transactions"].status == "failed"
    cov = {
        k.kind: k
        for k in read.coverage(engine, "demo", make_account_key("demo", 424242), False).kinds
    }
    assert cov["transactions"].state == "stale" and cov["transactions"].last_success_utc
    assert cov["transactions"].stored_records == 4 and cov["orders"].state == "complete"


def test_invalid_records_are_rejected_counted_and_not_stored(engine: Engine) -> None:
    pages = standard_pages()
    bad = tx("BAD", "DEPOSIT", 5.0)
    bad["amount"] = "not-a-number"
    nodate = tx("NODATE", "DEPOSIT", 5.0)
    del nodate["dateTime"]
    pages["transactions"][0] += [bad, nodate]
    res = _run(engine, FakeSource(pages))["transactions"]
    assert res.rejected == 2 and res.inserted == 4 and res.seen == 6
    assert res.reject_reasons == {"invalid:amount": 1, "missing:dateTime": 1}
    assert res.seen == res.inserted + res.duplicate + res.rejected + res.conflicting
    cov = {
        k.kind: k
        for k in read.coverage(engine, "demo", make_account_key("demo", 424242), False).kinds
    }
    assert cov["transactions"].rejected_records == 2
    assert any("rejected" in i for i in cov["transactions"].issues)


def test_changed_record_is_not_overwritten_and_is_reported(engine: Engine) -> None:
    pages = standard_pages()
    _run(engine, FakeSource(pages))
    changed = copy.deepcopy(pages)
    changed["dividends"][1] = [div("D1", 99.0, day=15)]
    res = _run(engine, FakeSource(changed), full=True)["dividends"]
    assert res.conflicting == 1 and res.inserted == 0
    rows, _ = read.list_rows(engine, "dividends", "demo", make_account_key("demo", 424242))
    assert sorted(r["amount"] for r in rows) == [3.2, 4.0]  # original kept


def test_record_repeated_on_two_pages_is_stored_once(engine: Engine) -> None:
    pages = standard_pages()
    pages["dividends"] = [[div("D2", 4.0), div("D1", 3.2)], [div("D1", 3.2)]]
    res = _run(engine, FakeSource(pages))["dividends"]
    assert res.inserted == 2 and res.duplicate == 1 and _count(engine, "dividends") == 2


def test_pagination_loop_and_empty_page_with_cursor_are_interruptions(engine: Engine) -> None:
    src = FakeSource(standard_pages())
    src.next_override[("orders", 1)] = "/api/v0/equity/history/orders?limit=50&cursor=1"
    res = _run(engine, src, kinds=("orders",))["orders"]
    assert res.status == "interrupted" and res.error_code == "pagination_loop"

    src2 = FakeSource({"orders": [[order(1)], []]})
    src2.next_override[("orders", 1)] = "/api/v0/equity/history/orders?limit=50&cursor=2"
    res = _run(engine, src2, kinds=("orders",))["orders"]
    assert res.status == "interrupted" and res.error_code == "empty_page_with_cursor"


def test_unsafe_next_page_link_interrupts_without_following(engine: Engine) -> None:
    from marketops.broker.t212 import Trading212ReadOnlyClient

    class Bad(Trading212ReadOnlyClient):
        def _get(self, path: str, params: dict[str, str] | None = None) -> object:
            return {"items": [tx("T1", "DEPOSIT", 1.0)], "nextPagePath": "https://evil.example/x"}

        def get_account_summary(self) -> dict[str, object]:  # type: ignore[override]
            return dict(FakeSource({}).summary)

    res = _run(engine, Bad("k", "s", "demo"), kinds=("transactions",))["transactions"]  # type: ignore[arg-type]
    assert res.status == "interrupted" and res.error_code == "pagination"
    assert res.inserted == 1


def test_run_left_running_by_a_dead_process_becomes_interrupted(engine: Engine) -> None:
    src = FakeSource(standard_pages())
    _run(engine, src, kinds=("orders",))
    with engine.begin() as c:
        c.execute(update(broker_sync_run).values(status="running", finished_utc=None))
    sync_account_history(engine, src, pacer=fast_pacer(), kinds=())
    with engine.connect() as c:
        row = c.execute(select(broker_sync_run.c.status, broker_sync_run.c.error_code)).one()
    assert tuple(row) == ("interrupted", "process_terminated")


def test_demo_and_live_data_never_mix(engine: Engine) -> None:
    demo = FakeSource(standard_pages(), environment="demo")
    live_pages = {"transactions": [[tx("T1", "DEPOSIT", 7.0)]]}
    live = FakeSource(live_pages, summary=dict(FakeSource({}).summary, id=999), environment="live")
    _run(engine, demo)
    _run(engine, live)
    demo_key, live_key = make_account_key("demo", 424242), make_account_key("live", 999)
    assert demo_key != live_key
    assert read.latest_account_key(engine, "demo") == demo_key
    assert read.latest_account_key(engine, "live") == live_key
    # "T1" exists in both scopes (same broker reference) without a uniqueness clash
    assert read.list_rows(engine, "transactions", "live", live_key)[1] == 1
    assert read.list_rows(engine, "transactions", "demo", demo_key)[1] == 4
    assert read.list_rows(engine, "transactions", "demo", live_key)[1] == 0


@pytest.mark.parametrize(
    "bad", [{}, {"id": None, "currency": "GBP"}, {"id": 1}, {"id": True, "currency": "GBP"}]
)
def test_unusable_account_summary_fails_closed_before_any_record_is_written(
    engine: Engine, bad: dict[str, object]
) -> None:
    with pytest.raises(BrokerError):
        sync_account_history(engine, FakeSource(standard_pages(), summary=bad), pacer=fast_pacer())
    with engine.connect() as c:
        assert c.execute(select(func.count()).select_from(broker_sync_run)).scalar_one() == 0


def test_only_one_sync_at_a_time(engine: Engine) -> None:
    assert sync._lock.acquire(blocking=False)
    try:
        with pytest.raises(SyncBusy):
            sync_account_history(engine, FakeSource(standard_pages()), pacer=fast_pacer())
    finally:
        sync._lock.release()
    assert not sync.sync_in_progress()

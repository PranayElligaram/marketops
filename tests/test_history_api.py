"""Private history routes: authentication, deliberate sync, filtering, totals, readiness."""

from __future__ import annotations

import copy
import logging
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from marketops.broker.t212 import BrokerError
from marketops.config import get_settings
from marketops.history import sync
from marketops.persistence import database
from marketops.persistence.models import broker_sync_run
from tests.history_fixtures import HDR, KEY, SECRET, SUMMARY, TOKEN, Env, order, tx

pytestmark = pytest.mark.filterwarnings("ignore:Dialect sqlite.*Decimal")

GETS = ["status", "summary", "readiness", "transactions", "orders", "dividends"]


# ------------------------------------------------------------------ authentication
@pytest.mark.parametrize("name", GETS)
def test_every_get_route_needs_the_token(env: Env, name: str) -> None:
    assert env.client.get(f"/history/{name}").status_code == 401
    assert (
        env.client.get(f"/history/{name}", headers={"X-MarketOps-Token": "wrong"}).status_code
        == 401
    )
    assert env.client.get(f"/history/{name}", headers=HDR).status_code in (200, 404)


def test_sync_needs_the_token_and_never_touches_the_broker_without_it(env: Env) -> None:
    assert env.client.post("/history/sync", json={}).status_code == 401
    assert (
        env.client.post(
            "/history/sync", json={}, headers={"X-MarketOps-Token": "x" * 20}
        ).status_code
        == 401
    )
    assert env.constructed == 0 and env.source.calls == []


def test_without_a_configured_token_everything_is_denied(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MARKETOPS_AGENT_API_TOKEN")
    get_settings.cache_clear()
    for name in GETS:
        assert env.client.get(f"/history/{name}", headers=HDR).status_code == 403
    assert env.client.post("/history/sync", json={}, headers=HDR).status_code == 403
    assert env.constructed == 0


def test_ordinary_reads_never_contact_the_broker(env: Env) -> None:
    for name in GETS:
        env.client.get(f"/history/{name}", headers=HDR)
    assert env.constructed == 0 and env.source.calls == []


# ------------------------------------------------------------------ sync action
def test_empty_state_is_explicit(env: Env) -> None:
    body = env.client.get("/history/status", headers=HDR).json()
    assert {k["state"] for k in body["kinds"]} == {"never_synced"}
    assert env.client.get("/history/orders", headers=HDR).json() == {"total": 0, "items": []}
    assert env.client.get("/history/summary", headers=HDR).status_code == 404


def test_deliberate_sync_stores_history_and_reports_status(env: Env) -> None:
    r = env.sync()
    assert (
        r.status_code == 202
        and r.json()["started"] is True
        and r.json()["account_currency"] == "GBP"
    )
    st = env.client.get("/history/status", headers=HDR).json()
    assert {k["kind"]: k["state"] for k in st["kinds"]} == {
        "orders": "complete",
        "transactions": "complete",
        "dividends": "complete",
    }
    assert (
        st["account_currency"] == "GBP" and st["environment"] == "demo" and not st["sync_running"]
    )
    assert {k["kind"]: k["stored_records"] for k in st["kinds"]} == {
        "orders": 4,
        "transactions": 4,
        "dividends": 2,
    }
    assert env.client.get("/history/transactions", headers=HDR).json()["total"] == 4


def test_sync_twice_does_not_duplicate(env: Env) -> None:
    env.sync()
    env.sync()
    env.sync(full=True)
    for kind, n in (("orders", 4), ("transactions", 4), ("dividends", 2)):
        assert env.client.get(f"/history/{kind}", headers=HDR).json()["total"] == n


def test_responses_hide_internal_and_personal_identifiers(env: Env) -> None:
    env.sync()
    texts = [env.client.get(f"/history/{n}", headers=HDR).text for n in GETS] + [env.sync().text]
    blob = "\n".join(texts)
    for forbidden in ("424242", "account_key", "content_hash", KEY, SECRET, TOKEN):
        assert forbidden not in blob


def test_live_requires_second_opt_in(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARKETOPS_T212_ENVIRONMENT", "live")
    get_settings.cache_clear()
    r = env.sync()
    assert r.status_code == 403 and "MARKETOPS_T212_ALLOW_LIVE" in r.json()["detail"]
    assert env.constructed == 0
    monkeypatch.setenv("MARKETOPS_T212_ALLOW_LIVE", "true")
    get_settings.cache_clear()
    assert env.sync().status_code == 202 and env.source.environment == "live"


def test_unconfigured_keys_and_missing_database(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARKETOPS_T212_API_KEY", "")
    get_settings.cache_clear()
    assert env.sync().status_code == 503 and env.constructed == 0
    monkeypatch.setenv("MARKETOPS_T212_API_KEY", KEY)
    get_settings.cache_clear()

    def boom() -> Engine:
        raise RuntimeError("no db")

    monkeypatch.setattr(database, "get_engine", boom)
    assert env.sync().status_code == 503
    assert env.client.get("/history/status", headers=HDR).status_code == 503


def test_second_sync_while_one_runs_is_refused(env: Env) -> None:
    assert sync._lock.acquire(blocking=False)
    try:
        assert env.sync().status_code == 409
        assert env.client.get("/history/status", headers=HDR).json()["sync_running"] is True
    finally:
        sync._lock.release()


@pytest.mark.parametrize(("code", "http"), [("auth", 502), ("scope", 502), ("rate_limit", 429)])
def test_account_lookup_failures_are_safe_and_write_nothing(env: Env, code: str, http: int) -> None:
    def fail() -> dict[str, Any]:
        raise BrokerError(f"safe message for {code}", code)

    env.source.get_account_summary = fail  # type: ignore[method-assign]
    r = env.sync()
    assert r.status_code == http and KEY not in r.text and SECRET not in r.text
    with env.engine.connect() as c:
        assert c.execute(select(func.count()).select_from(broker_sync_run)).scalar_one() == 0


def test_partial_sync_is_visible_through_the_api(env: Env) -> None:
    env.source.fail_at[("orders", 1)] = BrokerError("down", "unavailable")
    env.sync()
    kinds = {k["kind"]: k for k in env.client.get("/history/status", headers=HDR).json()["kinds"]}
    assert (
        kinds["orders"]["state"] == "partial"
        and kinds["orders"]["last_error_code"] == "unavailable"
    )
    assert kinds["orders"]["issues"]
    assert kinds["transactions"]["state"] == "complete"
    env.source.fail_at.clear()
    env.sync()
    kinds = {k["kind"]: k for k in env.client.get("/history/status", headers=HDR).json()["kinds"]}
    assert kinds["orders"]["state"] == "complete" and kinds["orders"]["stored_records"] == 4


def test_logs_contain_no_private_values(env: Env, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    env.sync()
    env.client.get("/history/summary", headers=HDR)
    text = caplog.text
    for forbidden in ("424242", KEY, SECRET, TOKEN, "AAA_US_EQ", "500.0", "T2"):
        assert forbidden not in text


# ------------------------------------------------------------------ filtering and totals
def test_date_filters_apply_to_lists_and_totals(env: Env) -> None:
    env.sync()
    q = "date_from=2024-02-10&date_to=2024-02-19"
    items = env.client.get(f"/history/transactions?{q}", headers=HDR).json()
    assert items["total"] == 2 and {i["source_id"] for i in items["items"]} == {"T2", "T3"}
    s = env.client.get(f"/history/summary?{q}", headers=HDR).json()
    assert s["deposits"] == [{"currency": "GBP", "amount": 500.0, "count": 1}]
    assert s["cash_fees"] == [{"currency": "GBP", "amount": 1.0, "count": 1}]
    assert s["withdrawals"] == []
    inclusive = env.client.get(
        "/history/transactions?date_from=2024-02-20&date_to=2024-02-20", headers=HDR
    )
    assert inclusive.json()["total"] == 1
    assert (
        env.client.get(
            "/history/orders?date_from=2024-03-01&date_to=2024-03-18", headers=HDR
        ).json()["total"]
        == 2
    )


def test_invalid_filters_are_422(env: Env) -> None:
    for q in (
        "date_from=2024-05-01&date_to=2024-01-01",
        "date_from=nope",
        "limit=0",
        "limit=501",
        "offset=-1",
    ):
        assert env.client.get(f"/history/transactions?{q}", headers=HDR).status_code == 422


def test_pagination_of_listing(env: Env) -> None:
    env.sync()
    first = env.client.get("/history/transactions?limit=3", headers=HDR).json()
    rest = env.client.get("/history/transactions?limit=3&offset=3", headers=HDR).json()
    assert (
        first["total"] == rest["total"] == 4
        and len(first["items"]) == 3
        and len(rest["items"]) == 1
    )
    assert [i["source_id"] for i in first["items"]] == ["T4", "T3", "T2"]  # newest first


def test_summary_totals_and_separation_of_cash_flows(env: Env) -> None:
    env.sync()
    s = env.client.get("/history/summary", headers=HDR).json()
    assert s["deposits"] == [{"currency": "GBP", "amount": 700.0, "count": 2}]
    assert s["withdrawals"] == [{"currency": "GBP", "amount": 50.0, "count": 1}]
    assert s["cash_fees"][0]["amount"] == 1.0 and s["interest"] == []
    assert s["dividends"] == [{"currency": "GBP", "amount": 7.2, "count": 2}]
    assert [m["month"] for m in s["dividends_by_month"]] == ["2024-04"]
    # buys/sells are order activity, never deposits or withdrawals
    assert (s["buy_count"], s["sell_count"]) == (2, 1)
    assert s["buy_value"][0]["amount"] == 200.0 and s["sell_value"][0]["amount"] == 120.0
    assert s["order_fees"] == [{"currency": "GBP", "amount": 0.5, "count": 1}]
    assert s["realised_pnl_reported"][0]["amount"] == 20.0
    assert s["unfilled_order_count"] == 1 and s["excluded_inconsistent"] == 0


def test_interest_transfers_and_unknown_types_are_not_flows(env: Env) -> None:
    env.source.pages["transactions"][0] += [
        tx("I1", "INTEREST_ON_FREE_CASH", 0.4, day=21),
        tx("X1", "TRANSFER", 1000.0, day=22),
        tx("U1", "BRAND_NEW", 9.0, day=23),
    ]
    env.sync()
    s = env.client.get("/history/summary", headers=HDR).json()
    assert s["interest"] == [{"currency": "GBP", "amount": 0.4, "count": 1}]
    assert s["transfers_count"] == 1 and s["unclassified_transactions"] == 1
    assert s["deposits"][0]["amount"] == 700.0 and s["withdrawals"][0]["amount"] == 50.0


def test_inconsistent_records_are_excluded_and_counted(env: Env) -> None:
    env.source.pages["transactions"][0].append(tx("BADDEP", "DEPOSIT", -75.0, day=22))
    env.source.pages["orders"][0].append(order(9, ccy="EUR", day=22))
    env.sync()
    s = env.client.get("/history/summary", headers=HDR).json()
    assert s["deposits"][0]["amount"] == 700.0  # the negative "deposit" is not netted in
    assert s["excluded_inconsistent"] == 2
    kinds = {k["kind"]: k for k in env.client.get("/history/status", headers=HDR).json()["kinds"]}
    assert kinds["transactions"]["inconsistent_records"] == 1
    assert any("inconsistent" in i for i in kinds["transactions"]["issues"])


def test_other_currency_flows_are_kept_separate_not_converted(env: Env) -> None:
    env.source.pages["transactions"][0].append(tx("EUR1", "DEPOSIT", 100.0, day=22, ccy="EUR"))
    env.sync()
    s = env.client.get("/history/summary", headers=HDR).json()
    assert {d["currency"]: d["amount"] for d in s["deposits"]} == {"GBP": 700.0, "EUR": 100.0}


# ------------------------------------------------------------------ performance readiness
def _ready(env: Env) -> dict[str, Any]:
    return {m["metric"]: m for m in env.client.get("/history/readiness", headers=HDR).json()}


def test_unsupported_performance_metrics_are_unavailable_and_valueless(env: Env) -> None:
    env.sync()
    r = _ready(env)
    for metric in (
        "money_weighted_return_xirr",
        "time_weighted_return",
        "portfolio_equity_curve",
        "total_return_pct",
        "benchmark_comparison",
    ):
        assert r[metric]["status"] == "unavailable" and r[metric]["value"] is None
        assert r[metric]["blockers"]


def test_net_flows_available_only_when_verifiable(env: Env) -> None:
    env.sync()
    flows = _ready(env)["net_external_flows"]
    assert flows["status"] == "available" and flows["blockers"] == []
    assert flows["value"] == {"deposits": 700.0, "withdrawals": 50.0, "net": 650.0}

    env.source.pages["transactions"][0].append(tx("X1", "TRANSFER", 10.0, day=22))
    env.sync(full=True)
    flows = _ready(env)["net_external_flows"]
    assert flows["status"] == "partial" and any("TRANSFER" in b for b in flows["blockers"])


def test_net_flows_blocked_by_incomplete_history(env: Env) -> None:
    env.source.fail_at[("transactions", 1)] = BrokerError("down", "unavailable")
    env.sync()
    flows = _ready(env)["net_external_flows"]
    assert flows["status"] == "partial" and any("not complete" in b for b in flows["blockers"])


def test_net_flows_blocked_by_foreign_currency_flows(env: Env) -> None:
    env.source.pages["transactions"][0].append(tx("EUR1", "DEPOSIT", 100.0, day=22, ccy="EUR"))
    env.sync()
    flows = _ready(env)["net_external_flows"]
    assert flows["status"] == "partial" and any(
        "other than the account currency" in b for b in flows["blockers"]
    )


def test_realised_pnl_is_available_only_when_it_reconciles(env: Env) -> None:
    env.sync()  # fixture summary says 12.5 while fills sum to 20.0
    pnl = _ready(env)["realised_pnl_reported"]
    assert pnl["status"] == "partial" and any("does not equal" in b for b in pnl["blockers"])

    summary = copy.deepcopy(SUMMARY)
    summary["investments"]["realizedProfitLoss"] = 20.0
    env.source.summary = summary
    env.sync(full=True)
    pnl = _ready(env)["realised_pnl_reported"]
    assert pnl["status"] == "available" and pnl["value"] == {
        "sum_of_fills": 20.0,
        "broker_all_time": 20.0,
    }


def test_readiness_before_any_sync_is_404_not_fabricated(env: Env) -> None:
    assert env.client.get("/history/readiness", headers=HDR).status_code == 404

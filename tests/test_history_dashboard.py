"""Streamlit Portfolio activity section driven end-to-end through the real FastAPI app."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import httpx
import pytest

from marketops.broker.t212 import BrokerError
from tests.history_fixtures import HDR, TOKEN, Env

pytestmark = pytest.mark.filterwarnings("ignore:Dialect sqlite.*Decimal")
DASHBOARD = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")
API = "http://localhost:8001"


def _app(env: Env, monkeypatch: pytest.MonkeyPatch) -> Any:
    from streamlit.testing.v1 import AppTest

    seen: list[tuple[str, str, dict[str, Any]]] = []

    def path(url: str) -> str:
        return url.removeprefix(API)

    def fake_get(url: str, **kw: Any) -> Any:
        seen.append(("GET", path(url), kw))
        if "/health" in url:
            return type("R", (), {"status_code": 200, "json": lambda self: {"status": "ready"}})()
        return env.client.get(path(url), params=kw.get("params"), headers=kw.get("headers"))

    def fake_post(url: str, **kw: Any) -> Any:
        seen.append(("POST", path(url), kw))
        return env.client.post(path(url), json=kw.get("json"), headers=kw.get("headers"))

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", fake_post)
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    at.seen = seen
    return at


def _load(at: Any, token: str | None = TOKEN) -> None:
    if token:
        at.text_input(key="tok_history").set_value(token).run()
    next(b for b in at.button if b.label == "Load saved history").click()
    at.run()


def _history_calls(at: Any) -> list[tuple[str, str]]:
    return [(m, p) for m, p, _ in at.seen if p.startswith("/history")]


def test_nothing_is_downloaded_or_even_read_on_page_load(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _app(env, monkeypatch)
    assert not at.exception
    assert _history_calls(at) == [] and env.constructed == 0
    assert not any(b.label == "Load saved history" and b.disabled for b in at.button)


def test_empty_history_shows_a_clear_missing_data_state(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _app(env, monkeypatch)
    _load(at)
    assert not at.exception and not at.error
    assert any("No account history has been imported yet" in i.value for i in at.info)
    assert env.constructed == 0  # loading saved history never contacts the broker


def test_wrong_token_is_a_clear_error(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(env, monkeypatch)
    _load(at, token="wrong-token")
    assert any(
        "Could not load status" in e.value and "Missing or invalid" in e.value for e in at.error
    )
    assert not any("Data completeness" in m.value for m in at.markdown)


def test_sync_requires_confirmation_then_posts_once_with_token(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _app(env, monkeypatch)
    at.text_input(key="tok_history").set_value(TOKEN).run()
    sync_btn = next(b for b in at.button if b.label == "Sync account history")
    assert sync_btn.disabled  # cannot be clicked without the explicit confirmation
    at.checkbox(key="act_confirm").check().run()
    next(b for b in at.button if b.label == "Sync account history").click()
    at.run()
    posts = [(m, p, kw) for m, p, kw in at.seen if m == "POST"]
    assert len(posts) == 1 and posts[0][1] == "/history/sync"
    assert posts[0][2]["headers"] == {"X-MarketOps-Token": TOKEN} and posts[0][2]["json"] == {
        "full": False
    }
    assert any("Sync started on the DEMO account" in s.value for s in at.success)
    assert env.constructed == 1


def test_sync_failure_is_shown_without_secrets(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> dict[str, Any]:
        raise BrokerError("Trading 212 rejected the API key or secret.", "auth")

    env.source.get_account_summary = fail  # type: ignore[method-assign]
    at = _app(env, monkeypatch)
    at.text_input(key="tok_history").set_value(TOKEN).run()
    at.checkbox(key="act_confirm").check().run()
    next(b for b in at.button if b.label == "Sync account history").click()
    at.run()
    assert any(
        "Could not start the sync" in e.value and "rejected the API key" in e.value
        for e in at.error
    )


def test_loaded_history_shows_overview_tables_and_completeness(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert env.sync().status_code == 202
    at = _app(env, monkeypatch)
    _load(at)
    assert not at.exception and not at.error
    assert any("DEMO TRADING 212 HISTORY" in s.value for s in at.success)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Deposits"] == "GBP 700.00" and metrics["Withdrawals"] == "GBP 50.00"
    assert metrics["Dividends received"] == "GBP 7.20" and metrics["Cash fees"] == "GBP 1.00"
    assert metrics["Buys (filled)"] == "2" and metrics["Sells (filled)"] == "1"
    completeness = at.dataframe[0].value
    assert set(completeness["State"]) == {"Complete"} and list(completeness["Records"]) == [4, 4, 2]
    assert len(at.dataframe) >= 5  # completeness + 3 record tables + (readiness is text)
    performance = " ".join(i.value for i in at.info)
    assert "time weighted return: unavailable" in performance
    assert "portfolio equity curve: unavailable" in performance
    assert any("net external flows" in s.value for s in at.success)
    text = " ".join(
        e.value for e in [*at.markdown, *at.caption, *at.info, *at.warning, *at.success]
    )
    assert "%" not in " ".join(i.value for i in at.info if "unavailable" in i.value)
    assert "424242" not in text


def test_partial_sync_is_flagged_not_hidden(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    env.source.fail_at[("transactions", 1)] = BrokerError("down", "unavailable")
    env.sync()
    at = _app(env, monkeypatch)
    _load(at)
    assert any("transactions:" in w.value and "incomplete" in w.value for w in at.warning)
    assert "Partial: may be missing records" in set(at.dataframe[0].value["State"])
    assert any("net external flows: unverified" in w.value for w in at.warning)


def test_date_filter_is_sent_and_changes_totals(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    env.sync()
    at = _app(env, monkeypatch)
    at.text_input(key="tok_history").set_value(TOKEN).run()
    at.date_input(key="act_from").set_value(dt.date(2024, 2, 10))
    at.date_input(key="act_to").set_value(dt.date(2024, 2, 19))
    next(b for b in at.button if b.label == "Load saved history").click()
    at.run()
    summary_call = next(kw for m, p, kw in at.seen if p == "/history/summary")
    assert summary_call["params"] == {"date_from": "2024-02-10", "date_to": "2024-02-19"}
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Deposits"] == "GBP 500.00" and metrics["Withdrawals"] == "none"


def test_token_is_sent_only_on_private_history_calls(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    env.sync()
    at = _app(env, monkeypatch)
    _load(at)
    for _method, path, kw in at.seen:
        if path.startswith("/history"):
            assert kw["headers"] == HDR
        else:
            assert not kw.get("headers")

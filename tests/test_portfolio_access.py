"""Access control for real brokerage data. No Trading 212 credentials are needed."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from api.main import app
from api.routes.portfolio import clear_cache
from api.routes.portfolio import router as portfolio_router
from api.security import private_access
from marketops.agents.runner import run_research
from marketops.config import Settings, get_settings
from marketops.portfolio.analytics import (
    EXAMPLE_ACCOUNT_SUMMARY,
    EXAMPLE_POSITIONS,
    summarise_positions,
)

TOKEN = "test-token-0123456789abcdef"
KEY, SECRET = "key-SENTINEL-1", "secret-SENTINEL-2"
DASHBOARD = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")
HDR = {"X-MarketOps-Token": TOKEN}


class _FakeBroker:
    def __init__(self, **_: Any) -> None: ...

    def get_positions(self) -> list[dict[str, Any]]:
        return EXAMPLE_POSITIONS

    def get_account_summary(self) -> dict[str, Any]:
        return EXAMPLE_ACCOUNT_SUMMARY


def _env(mp: pytest.MonkeyPatch, token: str | None, broker: bool = True) -> None:
    mp.delenv("MARKETOPS_AGENT_API_TOKEN", raising=False)
    if token:
        mp.setenv("MARKETOPS_AGENT_API_TOKEN", token)
    mp.setenv("MARKETOPS_T212_API_KEY", KEY if broker else "")
    mp.setenv("MARKETOPS_T212_API_SECRET", SECRET if broker else "")
    get_settings.cache_clear()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr("api.routes.portfolio.Trading212ReadOnlyClient", _FakeBroker)
    clear_cache()
    yield TestClient(app)
    clear_cache()
    get_settings.cache_clear()


def test_overview_denied_without_token(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, TOKEN)
    r = client.get("/portfolio/overview")
    assert r.status_code == 401 and "holdings" not in r.text


def test_overview_denied_with_wrong_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch, TOKEN)
    r = client.get("/portfolio/overview", headers={"X-MarketOps-Token": TOKEN[:-1] + "x"})
    assert r.status_code == 401
    assert TOKEN not in r.text and KEY not in r.text and SECRET not in r.text


def test_overview_denied_when_no_token_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch, None)
    for h in ({}, {"X-MarketOps-Token": ""}, {"X-MarketOps-Token": "x"}):
        assert client.get("/portfolio/overview", headers=h).status_code == 403


def test_overview_denied_when_token_too_short(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch, "short")
    r = client.get("/portfolio/overview", headers={"X-MarketOps-Token": "short"})
    assert r.status_code == 403


def test_overview_allowed_with_correct_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch, TOKEN)
    r = client.get("/portfolio/overview", headers=HDR)
    assert r.status_code == 200 and r.json()["source"] == "trading212"


def test_authorised_but_unconfigured_broker_is_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch, TOKEN, broker=False)
    r = client.get("/portfolio/overview", headers=HDR)
    assert r.status_code == 503 and TOKEN not in r.text


def test_broker_error_response_and_logs_have_no_secrets(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _env(monkeypatch, TOKEN)
    transport = httpx.MockTransport(lambda req: httpx.Response(401, json={"x": KEY + SECRET}))
    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=transport, **kw))
    with caplog.at_level(logging.DEBUG):
        r = TestClient(app).get("/portfolio/overview", headers=HDR)
    assert r.status_code == 502
    blob = r.text + caplog.text
    assert KEY not in blob and SECRET not in blob and TOKEN not in blob
    get_settings.cache_clear()


def test_synthetic_example_needs_no_token_or_credentials(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch, None, broker=False)
    r = client.get("/portfolio/example")
    assert r.status_code == 200 and r.json()["source"] == "synthetic_example"


def test_every_non_example_portfolio_route_is_guarded() -> None:
    routes = [r for r in portfolio_router.routes if isinstance(r, APIRoute)]
    assert any(r.path == "/portfolio/overview" for r in routes)
    for route in routes:
        if route.path != "/portfolio/example":
            deps = [d.call for d in route.dependant.dependencies]
            assert private_access in deps, route.path


def test_agent_audit_withholds_broker_question(monkeypatch: pytest.MonkeyPatch) -> None:
    s = Settings(
        agent_allow_broker_portfolio=True,
        t212_api_key=KEY,
        t212_api_secret=SECRET,
        agent_api_token=TOKEN,
    )
    monkeypatch.setattr("marketops.agents.tools.Trading212ReadOnlyClient", _FakeBroker)
    _, audit = run_research("I hold 500 shares of AAPL, concentration?", "trading212", s, None)
    assert audit["question_preview"] == "withheld"
    blob = str(audit)
    assert KEY not in blob and SECRET not in blob and TOKEN not in blob
    _, ex = run_research("hello AAPL?", "example", s, None)
    assert ex["question_preview"].startswith("hello")


class _Resp:
    def __init__(self, data: Any, code: int = 200) -> None:
        self.data, self.status_code = data, code

    def json(self) -> Any:
        return self.data


_EXAMPLE = json.loads(summarise_positions([], source="synthetic_example").model_dump_json())


def _dash(monkeypatch: pytest.MonkeyPatch, seen: list[dict[str, Any]]) -> Any:
    from streamlit.testing.v1 import AppTest

    def fake_get(url: str, **kw: Any) -> _Resp:
        seen.append({"url": url, "headers": kw.get("headers")})
        if "health" in url:
            return _Resp({"status": "ready"})
        if url.endswith("/portfolio/example"):
            return _Resp(_EXAMPLE)
        if url.endswith("/portfolio/overview"):
            return _Resp({"detail": "Missing or invalid access token."}, 401)
        return _Resp({"detail": "n/a"}, 503)

    monkeypatch.setattr(httpx, "get", fake_get)
    return AppTest.from_file(DASHBOARD, default_timeout=30)


def _click(at: Any, label: str) -> None:
    next(b for b in at.button if b.label == label).click()
    at.run()


def test_dashboard_synthetic_mode_sends_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARKETOPS_AGENT_API_TOKEN", TOKEN)
    seen: list[dict[str, Any]] = []
    at = _dash(monkeypatch, seen).run()
    _click(at, "Load portfolio")
    assert not at.exception
    ex = [s for s in seen if s["url"].endswith("/portfolio/example")]
    assert ex and all(not s["headers"] for s in ex)


def test_dashboard_ignores_environment_token_for_real_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MARKETOPS_AGENT_API_TOKEN", TOKEN)
    seen: list[dict[str, Any]] = []
    at = _dash(monkeypatch, seen).run()
    next(r for r in at.radio if r.label == "Portfolio data").set_value("Trading 212 account").run()
    _click(at, "Load portfolio")
    ov = [s for s in seen if s["url"].endswith("/portfolio/overview")]
    assert ov and all(not s["headers"] for s in ov)
    assert any("Missing or invalid" in e.value for e in at.error)


def test_dashboard_sends_user_typed_token_for_real_data(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []
    at = _dash(monkeypatch, seen).run()
    next(r for r in at.radio if r.label == "Portfolio data").set_value("Trading 212 account").run()
    next(t for t in at.text_input if t.key == "tok_portfolio").set_value(TOKEN).run()
    _click(at, "Load portfolio")
    ov = [s for s in seen if s["url"].endswith("/portfolio/overview")]
    assert ov and ov[-1]["headers"] == HDR

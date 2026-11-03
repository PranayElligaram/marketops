"""Offline Trading 212 contract, analytics, route and dashboard tests (no credentials/network).

Fixtures follow docs.trading212.com: GET /equity/positions and GET /equity/account/summary.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes.portfolio import clear_cache
from marketops.broker.t212 import (
    BrokerError,
    Trading212ReadOnlyClient,
    require_environment_allowed,
)
from marketops.config import Settings, get_settings
from marketops.portfolio.analytics import summarise_positions

KEY, SECRET, TOKEN = "k-SENTINEL-key", "s-SENTINEL-secret", "tok-0123456789abcdef-SENTINEL"
HDR = {"X-MarketOps-Token": TOKEN}
DASHBOARD = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")


def _position(
    ticker: str, name: str, isin: str, ccy: str, value: float, cost: float, fx: float | None
) -> dict[str, Any]:
    return {
        "averagePricePaid": 1.0,
        "createdAt": "2024-03-01T10:00:00Z",
        "currentPrice": 2.0,
        "instrument": {"currency": ccy, "isin": isin, "name": name, "ticker": ticker},
        "quantity": 10.5,
        "quantityAvailableForTrading": 10.5,
        "quantityInPies": 0,
        "walletImpact": {
            "currency": "EUR",
            "currentValue": value,
            "fxImpact": fx,
            "totalCost": cost,
            "unrealizedProfitLoss": value - cost,
        },
    }


POSITIONS: list[dict[str, Any]] = [
    _position("AAPL_US_EQ", "Apple", "US0378331005", "USD", 1900.0, 1500.0, -12.5),
    _position("SAPd_EQ", "SAP", "DE0007164600", "EUR", 900.0, 1000.0, None),
    _position("IWDAe_EQ", "World ETF", "IE00B4L5Y983", "EUR", 1200.0, 1000.0, None),
]
SUMMARY: dict[str, Any] = {
    "id": 123,
    "currency": "EUR",
    "totalValue": 4500.0,
    "cash": {"availableToTrade": 400.0, "inPies": 50.0, "reservedForOrders": 50.0},
    "investments": {
        "currentValue": 4000.0,
        "realizedProfitLoss": 75.0,
        "totalCost": 3500.0,
        "unrealizedProfitLoss": 500.0,
    },
}

Handler = Callable[[httpx.Request], httpx.Response]
_REAL_CLIENT = httpx.Client


def _transport(monkeypatch: pytest.MonkeyPatch, handler: Handler) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    def factory(*a: Any, **kw: Any) -> httpx.Client:
        return _REAL_CLIENT(*a, transport=httpx.MockTransport(wrapped), **kw)

    monkeypatch.setattr(httpx, "Client", factory)
    return seen


def _ok(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/equity/positions"):
        return httpx.Response(200, json=POSITIONS)
    return httpx.Response(200, json=SUMMARY)


# ------------------------------------------------------------------ adapter contract
def test_client_uses_basic_auth_get_only_and_documented_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _transport(monkeypatch, _ok)
    c = Trading212ReadOnlyClient(KEY, SECRET, "demo")
    assert len(c.get_positions()) == 3 and c.get_account_summary()["currency"] == "EUR"
    assert [r.method for r in seen] == ["GET", "GET"]
    assert [str(r.url) for r in seen] == [
        "https://demo.trading212.com/api/v0/equity/positions",
        "https://demo.trading212.com/api/v0/equity/account/summary",
    ]
    expected = "Basic " + base64.b64encode(f"{KEY}:{SECRET}".encode()).decode()
    assert all(r.headers["authorization"] == expected for r in seen)


def test_client_has_no_write_surface() -> None:
    c = Trading212ReadOnlyClient(KEY, SECRET, "demo")
    public = {n for n in dir(c) if not n.startswith("_")}
    assert public == {
        "BASE_URLS",
        "base_url",
        "environment",
        "get_account_summary",
        "get_history_page",
        "get_positions",
    }
    assert KEY not in repr(c) and SECRET not in repr(c)


def test_live_needs_explicit_opt_in_and_default_is_demo() -> None:
    assert Settings(_env_file=None).t212_environment == "demo"
    assert Settings(_env_file=None).t212_allow_live is False
    with pytest.raises(BrokerError, match="Live Trading 212 access is disabled"):
        require_environment_allowed("live", False)
    require_environment_allowed("live", True)
    require_environment_allowed("demo", False)
    with pytest.raises(BrokerError):
        Trading212ReadOnlyClient(KEY, SECRET, "prod")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("status", "text"),
    [
        (401, "rejected the API key"),
        (403, "lacks a required read scope"),
        (408, "timed out"),
        (429, "rate limit"),
        (503, "unavailable"),
        (404, "HTTP 404"),
    ],
)
def test_http_errors_map_to_safe_messages(
    monkeypatch: pytest.MonkeyPatch, status: int, text: str
) -> None:
    _transport(monkeypatch, lambda r: httpx.Response(status, text=f"echo {KEY} {SECRET}"))
    with pytest.raises(BrokerError, match=text) as exc:
        Trading212ReadOnlyClient(KEY, SECRET, "demo").get_positions()
    assert KEY not in str(exc.value) and SECRET not in str(exc.value)


def test_rate_limit_reset_header_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    _transport(
        monkeypatch, lambda r: httpx.Response(429, headers={"x-ratelimit-reset": "<script>"})
    )
    with pytest.raises(BrokerError) as exc:
        Trading212ReadOnlyClient(KEY, SECRET, "demo").get_positions()
    assert "script" not in str(exc.value)


def test_network_failure_and_bad_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = Trading212ReadOnlyClient(KEY, SECRET, "demo")
    _transport(monkeypatch, boom)
    with pytest.raises(BrokerError, match="unreachable"):
        client.get_positions()
    _transport(monkeypatch, lambda r: httpx.Response(200, text="not json"))
    with pytest.raises(BrokerError, match="invalid data"):
        client.get_positions()
    _transport(monkeypatch, lambda r: httpx.Response(200, json={"items": []}))
    with pytest.raises(BrokerError, match="unexpected positions"):
        client.get_positions()
    _transport(monkeypatch, lambda r: httpx.Response(200, json=[]))
    with pytest.raises(BrokerError, match="unexpected account summary"):
        client.get_account_summary()


def test_redirects_are_not_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _transport(
        monkeypatch, lambda r: httpx.Response(302, headers={"location": "https://evil.example/"})
    )
    with pytest.raises(BrokerError):
        Trading212ReadOnlyClient(KEY, SECRET, "demo").get_positions()
    assert len(seen) == 1


# ------------------------------------------------------------------ analytics
def test_overview_from_documented_fixture() -> None:
    ov = summarise_positions(POSITIONS, source="trading212", summary=SUMMARY)
    assert ov.account_currency == "EUR"
    assert ov.holdings_value == 4000 and ov.holdings_cost == 3500 and ov.unrealised_pnl == 500
    assert ov.unrealised_pnl_pct == pytest.approx(14.29, abs=0.01)
    assert [h.ticker for h in ov.holdings] == ["AAPL_US_EQ", "IWDAe_EQ", "SAPd_EQ"]
    assert [h.weight_pct for h in ov.holdings] == [47.5, 30.0, 22.5]
    assert ov.holdings[0].unrealised_pnl_pct == pytest.approx(26.67, abs=0.01)
    assert ov.holdings[2].unrealised_pnl_pct == -10.0
    assert ov.cash and ov.cash.total_cash == 500 and ov.portfolio_value_incl_cash == 4500
    assert ov.cash_pct_of_portfolio == pytest.approx(11.11, abs=0.01)
    assert ov.broker_reported_total_value == 4500 and ov.broker_realised_pnl_all_time == 75
    assert ov.total_fx_impact == -12.5
    assert not ov.performance.available
    assert ov.data_quality_warnings == []


def test_quote_currency_prices_do_not_affect_weights() -> None:
    rows = copy.deepcopy(POSITIONS)
    rows[1]["currentPrice"] = 1_000_000
    ov = summarise_positions(rows, source="trading212")
    assert ov.holdings_value == 4000 and ov.holdings[0].weight_pct == 47.5


def test_concentration_and_exposure() -> None:
    ov = summarise_positions(POSITIONS, source="trading212")
    c = ov.concentration
    assert c.largest_ticker == "AAPL_US_EQ" and c.largest_weight_pct == 47.5
    assert c.top3_weight_pct == 100 and c.holdings_count == 3
    assert c.herfindahl_index == pytest.approx(0.475**2 + 0.3**2 + 0.225**2, abs=1e-4)
    assert c.effective_number_of_holdings == pytest.approx(1 / c.herfindahl_index, abs=0.01)
    assert any("single-position" in w for w in c.warnings)
    assert any("Only 3 holding" in w for w in c.warnings)
    ccy = {e.key: e.weight_pct for e in ov.currency_exposure}
    assert ccy == {"USD": 47.5, "EUR": 52.5}
    assert {e.key for e in ov.isin_country_exposure} == {"US", "DE", "IE"}
    assert ov.cash is None
    assert any("Account summary unavailable" in w for w in ov.data_quality_warnings)


def test_empty_portfolio_is_valid_not_fabricated() -> None:
    ov = summarise_positions([], source="trading212", summary=None)
    assert ov.holdings == [] and ov.holdings_value == 0 and ov.unrealised_pnl_pct is None
    assert ov.concentration.largest_ticker is None and ov.concentration.warnings == []


def test_zero_cost_has_no_percentage() -> None:
    rows = copy.deepcopy(POSITIONS)
    rows[0]["walletImpact"].update(totalCost=0.0, unrealizedProfitLoss=1900.0)
    assert summarise_positions(rows, source="trading212").holdings[0].unrealised_pnl_pct is None


def test_missing_instrument_currency_is_unknown_not_guessed() -> None:
    rows = copy.deepcopy(POSITIONS)
    del rows[1]["instrument"]["currency"]
    ov = summarise_positions(rows, source="trading212")
    assert "Unknown" in {e.key for e in ov.currency_exposure}


def test_summary_currency_mismatch_drops_cash_not_positions() -> None:
    ov = summarise_positions(POSITIONS, source="trading212", summary=dict(SUMMARY, currency="GBP"))
    assert ov.cash is None and ov.portfolio_value_incl_cash is None
    assert any("unusable" in w for w in ov.data_quality_warnings)
    assert ov.holdings_value == 4000


def test_summary_missing_fields_and_stale_mismatch_warned() -> None:
    ov = summarise_positions(POSITIONS, source="trading212", summary={"currency": "EUR"})
    assert ov.cash is None
    drift = copy.deepcopy(SUMMARY)
    drift["investments"]["currentValue"] = 4100.0
    ov = summarise_positions(POSITIONS, source="trading212", summary=drift)
    assert any("differs from the sum" in w for w in ov.data_quality_warnings)


def test_broker_pnl_inconsistency_flagged() -> None:
    rows = copy.deepcopy(POSITIONS)
    rows[0]["walletImpact"]["unrealizedProfitLoss"] = 999.0
    ov = summarise_positions(rows, source="trading212")
    assert any("AAPL_US_EQ" in w for w in ov.data_quality_warnings)


def test_no_history_means_no_performance_numbers() -> None:
    dumped = summarise_positions(POSITIONS, source="trading212").model_dump()
    assert dumped["performance"]["available"] is False
    assert not any(k in dumped for k in ("twr", "mwr", "return_pct", "cagr"))


# ------------------------------------------------------------------ API
class _Broker:
    positions: Any = POSITIONS
    summary: Any = SUMMARY
    calls = 0

    def __init__(self, **_: Any) -> None: ...

    def get_positions(self) -> list[dict[str, Any]]:
        type(self).calls += 1
        if isinstance(self.positions, Exception):
            raise self.positions
        return self.positions  # type: ignore[no-any-return]

    def get_account_summary(self) -> dict[str, Any]:
        if isinstance(self.summary, Exception):
            raise self.summary
        return self.summary  # type: ignore[no-any-return]


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    for k, v in {
        "MARKETOPS_AGENT_API_TOKEN": TOKEN,
        "MARKETOPS_T212_API_KEY": KEY,
        "MARKETOPS_T212_API_SECRET": SECRET,
        "MARKETOPS_T212_ENVIRONMENT": "demo",
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("MARKETOPS_T212_ALLOW_LIVE", raising=False)
    _Broker.positions, _Broker.summary, _Broker.calls = POSITIONS, SUMMARY, 0
    monkeypatch.setattr("api.routes.portfolio.Trading212ReadOnlyClient", _Broker)
    get_settings.cache_clear()
    clear_cache()
    return TestClient(app)


def test_overview_labels_demo_environment(api: TestClient) -> None:
    body = api.get("/portfolio/overview", headers=HDR).json()
    assert body["source"] == "trading212" and body["broker_environment"] == "demo"
    assert body["cash"]["total_cash"] == 500
    dumped = json.dumps(body)
    assert KEY not in dumped and SECRET not in dumped and TOKEN not in dumped


def test_overview_unauthorised_never_calls_broker(api: TestClient) -> None:
    assert api.get("/portfolio/overview").status_code == 401
    wrong = {"X-MarketOps-Token": "x" * 20}
    assert api.get("/portfolio/overview", headers=wrong).status_code == 401
    assert _Broker.calls == 0


def test_live_blocked_without_opt_in(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARKETOPS_T212_ENVIRONMENT", "live")
    get_settings.cache_clear()
    r = api.get("/portfolio/overview", headers=HDR)
    assert r.status_code == 403 and "disabled" in r.json()["detail"] and _Broker.calls == 0
    monkeypatch.setenv("MARKETOPS_T212_ALLOW_LIVE", "true")
    get_settings.cache_clear()
    clear_cache()
    r = api.get("/portfolio/overview", headers=HDR)
    assert r.status_code == 200 and r.json()["broker_environment"] == "live"


def test_rate_limit_auth_and_outage_mapping(api: TestClient) -> None:
    for msg, code in [
        ("Trading 212 rate limit reached. Retry later.", 429),
        ("Trading 212 rejected the API key or secret.", 502),
        ("Trading 212 is unavailable (HTTP 503).", 502),
    ]:
        clear_cache()
        _Broker.positions = BrokerError(msg)
        r = api.get("/portfolio/overview", headers=HDR)
        assert r.status_code == code and r.json()["detail"] == msg


def test_summary_failure_degrades_to_positions_only(api: TestClient) -> None:
    _Broker.summary = BrokerError("Trading 212 rate limit reached. Retry later.")
    body = api.get("/portfolio/overview", headers=HDR).json()
    assert body["cash"] is None and body["holdings_value"] == 4000


def test_snapshot_cache_avoids_burst_requests(api: TestClient) -> None:
    api.get("/portfolio/overview", headers=HDR)
    api.get("/portfolio/overview", headers=HDR)
    assert _Broker.calls == 1


def test_errors_are_not_cached(api: TestClient) -> None:
    _Broker.positions = BrokerError("Trading 212 is unavailable (HTTP 503).")
    assert api.get("/portfolio/overview", headers=HDR).status_code == 502
    _Broker.positions = POSITIONS
    assert api.get("/portfolio/overview", headers=HDR).status_code == 200


def test_incompatible_data_gives_no_totals(api: TestClient) -> None:
    _Broker.positions = [{"instrument": {"ticker": "X"}, "walletImpact": {}}]
    r = api.get("/portfolio/overview", headers=HDR)
    assert r.status_code == 502 and "No portfolio totals" in r.json()["detail"]


def test_secrets_absent_from_logs_on_failure(
    api: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    _Broker.positions = BrokerError("Trading 212 is unavailable (HTTP 503).")
    api.get("/portfolio/overview", headers=HDR)
    api.get("/portfolio/overview")
    assert KEY not in caplog.text and SECRET not in caplog.text and TOKEN not in caplog.text


def test_example_route_is_synthetic_and_complete() -> None:
    body = TestClient(app).get("/portfolio/example").json()
    assert body["source"] == "synthetic_example" and body["broker_environment"] is None
    assert body["cash"]["total_cash"] == 250 and body["performance"]["available"] is False


# ------------------------------------------------------------------ dashboard
class _Resp:
    def __init__(self, data: Any, code: int = 200) -> None:
        self.data, self.status_code = data, code

    def json(self) -> Any:
        return self.data


def _dash(monkeypatch: pytest.MonkeyPatch, response: _Resp) -> Any:
    from streamlit.testing.v1 import AppTest

    seen: list[tuple[str, dict[str, Any]]] = []

    def fake_get(url: str, **kw: Any) -> _Resp:
        seen.append((url, kw))
        if "health" in url:
            return _Resp({"status": "ready"})
        return response if "/portfolio/" in url else _Resp({"detail": "n/a"}, 503)

    monkeypatch.setattr(httpx, "get", fake_get)
    at = AppTest.from_file(DASHBOARD, default_timeout=30).run()
    at.seen = seen
    return at


def _load(at: Any, broker: bool) -> None:
    if broker:
        at.radio[0].set_value("Trading 212 account").run()
    next(b for b in at.button if b.label == "Load portfolio").click()
    at.run()


def _broker_body(env: str) -> dict[str, Any]:
    ov = summarise_positions(POSITIONS, source="trading212", summary=SUMMARY)
    ov.broker_environment = env  # type: ignore[assignment]
    return json.loads(ov.model_dump_json())  # type: ignore[no-any-return]


def test_dashboard_renders_synthetic_portfolio_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    at = _dash(monkeypatch, _Resp(TestClient(app).get("/portfolio/example").json()))
    _load(at, broker=False)
    assert not at.exception
    assert any("SYNTHETIC EXAMPLE" in w.value for w in at.warning)
    assert any("Historical performance: not available" in i.value for i in at.info)
    _, kw = next(s for s in at.seen if "/portfolio/example" in s[0])
    assert not kw.get("headers")  # synthetic request carries no token
    assert any(m.label == "Cash" for m in at.metric) and at.dataframe


def test_dashboard_renders_demo_broker_data_and_sends_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    at = _dash(monkeypatch, _Resp(_broker_body("demo")))
    at.radio[0].set_value("Trading 212 account").run()
    at.text_input(key="tok_portfolio").set_value(TOKEN).run()
    _load(at, broker=False)
    assert not at.exception
    assert any("DEMO TRADING 212 DATA" in s.value for s in at.success)
    assert any("single-position" in w.value for w in at.warning)
    assert any("Account currency: **EUR**" in c.value for c in at.caption)
    _, kw = next(s for s in at.seen if "/portfolio/overview" in s[0])
    assert kw["headers"] == {"X-MarketOps-Token": TOKEN}


def test_dashboard_marks_live_data_distinctly(monkeypatch: pytest.MonkeyPatch) -> None:
    at = _dash(monkeypatch, _Resp(_broker_body("live")))
    _load(at, broker=True)
    assert any("LIVE TRADING 212 DATA" in e.value for e in at.error)


def test_dashboard_empty_account_and_error_states(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = json.loads(summarise_positions([], source="trading212").model_dump_json())
    at = _dash(monkeypatch, _Resp(empty))
    _load(at, broker=True)
    assert not at.exception and any("no open positions" in i.value for i in at.info)

    at = _dash(monkeypatch, _Resp({"detail": "Trading 212 rate limit reached."}, 429))
    _load(at, broker=True)
    assert not at.exception
    assert any("rate limit" in e.value for e in at.error)


# ------------------------------------------------------------------ final review regressions
def test_empty_positions_with_cash_uses_summary_currency() -> None:
    summary = {
        "currency": "GBP",
        "totalValue": 1000.0,
        "cash": {"availableToTrade": 1000.0, "inPies": 0, "reservedForOrders": 0},
        "investments": {
            "currentValue": 0,
            "realizedProfitLoss": 0,
            "totalCost": 0,
            "unrealizedProfitLoss": 0,
        },
    }
    ov = summarise_positions([], source="trading212", summary=summary)
    assert ov.account_currency == "GBP" and ov.holdings == []
    assert ov.cash and ov.cash.total_cash == 1000
    assert ov.portfolio_value_incl_cash == 1000 and ov.cash_pct_of_portfolio == 100
    assert ov.data_quality_warnings == []


def test_total_value_excluding_cash_is_not_double_counted() -> None:
    summary = copy.deepcopy(SUMMARY)
    summary["totalValue"] = 4000.0  # investments only
    ov = summarise_positions(POSITIONS, source="trading212", summary=summary)
    assert ov.portfolio_value_incl_cash is None and ov.cash_pct_of_portfolio is None
    assert ov.cash is not None and ov.broker_reported_total_value == 4000
    assert any("not included" in w for w in ov.data_quality_warnings)


def test_unreconciled_total_value_hides_combined_total() -> None:
    summary = copy.deepcopy(SUMMARY)
    summary["totalValue"] = 9999.0
    ov = summarise_positions(POSITIONS, source="trading212", summary=summary)
    assert ov.portfolio_value_incl_cash is None
    assert any("does not reconcile" in w for w in ov.data_quality_warnings)


def test_malformed_cash_value_drops_cash_only() -> None:
    summary = copy.deepcopy(SUMMARY)
    summary["cash"]["inPies"] = "n/a"
    ov = summarise_positions(POSITIONS, source="trading212", summary=summary)
    assert ov.cash is None and ov.holdings_value == 4000

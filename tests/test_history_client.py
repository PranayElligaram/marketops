"""History endpoints of the read-only client: auth, pagination safety, errors, rate limits."""

from __future__ import annotations

import base64
import inspect
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from marketops.broker import t212
from marketops.broker.t212 import (
    BrokerError,
    HistoryPacer,
    Trading212ReadOnlyClient,
    validate_next_page_path,
)

KEY, SECRET = "k-SENTINEL-key", "s-SENTINEL-secret"
_REAL = httpx.Client
Handler = Callable[[httpx.Request], httpx.Response]
BASE = "/api/v0/equity/history"


def _net(monkeypatch: pytest.MonkeyPatch, handler: Handler) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    monkeypatch.setattr(
        httpx, "Client", lambda *a, **kw: _REAL(*a, transport=httpx.MockTransport(wrapped), **kw)
    )
    return seen


def _client() -> Trading212ReadOnlyClient:
    return Trading212ReadOnlyClient(KEY, SECRET, "live")


def _pacer() -> tuple[HistoryPacer, list[float]]:
    sleeps: list[float] = []
    return HistoryPacer(0, sleep=sleeps.append), sleeps


# ------------------------------------------------------------------ request shape
@pytest.mark.parametrize("kind", ["orders", "transactions", "dividends"])
def test_first_page_is_a_basic_auth_get_on_the_documented_path(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    seen = _net(
        monkeypatch, lambda r: httpx.Response(200, json={"items": [], "nextPagePath": None})
    )
    items, nxt = _client().get_history_page(kind, pacer=_pacer()[0])  # type: ignore[arg-type]
    assert items == [] and nxt is None
    (req,) = seen
    assert req.method == "GET"
    assert str(req.url) == f"https://live.trading212.com{BASE}/{kind}?limit=50"
    assert (
        req.headers["authorization"]
        == "Basic " + base64.b64encode(f"{KEY}:{SECRET}".encode()).decode()
    )


def test_pagination_follows_validated_next_page_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        cursor = r.url.params.get("cursor")
        if cursor is None:
            return httpx.Response(
                200,
                json={
                    "items": [{"id": 1}],
                    "nextPagePath": f"{BASE}/orders?limit=50&cursor=1760346100000",
                },
            )
        return httpx.Response(200, json={"items": [{"id": 2}], "nextPagePath": None})

    seen = _net(monkeypatch, handler)
    c, p = _client(), _pacer()[0]
    items, nxt = c.get_history_page("orders", pacer=p)
    assert nxt == f"{BASE}/orders?limit=50&cursor=1760346100000"
    items2, nxt2 = c.get_history_page("orders", nxt, pacer=p)
    assert items == [{"id": 1}] and items2 == [{"id": 2}] and nxt2 is None
    assert [r.url.host for r in seen] == ["live.trading212.com"] * 2
    assert seen[1].url.params["cursor"] == "1760346100000" and seen[1].url.params["limit"] == "50"


@pytest.mark.parametrize(
    "bad",
    [
        "https://evil.example/api/v0/equity/history/orders?limit=50&cursor=1",
        "//evil.example/api/v0/equity/history/orders?cursor=1",
        "http://live.trading212.com/api/v0/equity/history/orders?cursor=1",
        f"{BASE}/transactions?limit=50&cursor=1",  # a different endpoint
        "/api/v0/equity/orders?cursor=1",
        "/api/v0/equity/history/orders/../../account/summary",
        f"{BASE}/orders?cursor=1&cursor=2",
        f"{BASE}/orders?cursor=1&url=http://evil",
        f"{BASE}/orders?cursor=../../x",
        f"{BASE}/orders?cursor=1%0d%0aX-Evil:1",
        f"{BASE}/orders?limit=5000&cursor=1",
        f"{BASE}/orders?limit=abc",
        f"{BASE}/orders?cursor=1#frag",
        f"{BASE}/orders?cursor=1 2",
        f"{BASE}\\orders?cursor=1",
        "user@evil.example" + BASE + "/orders",
        "",
        "x" * 600,
        123,
    ],
)
def test_unsafe_or_foreign_next_page_paths_are_refused_without_any_request(
    monkeypatch: pytest.MonkeyPatch, bad: Any
) -> None:
    seen = _net(monkeypatch, lambda r: httpx.Response(200, json={"items": []}))
    with pytest.raises(BrokerError) as exc:
        _client().get_history_page("orders", bad, pacer=_pacer()[0])
    assert exc.value.code == "pagination" and seen == []


def test_valid_next_page_paths_are_accepted() -> None:
    assert validate_next_page_path("orders", f"{BASE}/orders?limit=2&cursor=1760346100000") == {
        "limit": "2",
        "cursor": "1760346100000",
    }
    assert validate_next_page_path(
        "transactions", f"{BASE}/transactions?cursor=2024-03-01T10:00:00Z".replace("Z", "Z")
    ) == {"cursor": "2024-03-01T10:00:00Z"}
    assert validate_next_page_path("dividends", f"{BASE}/dividends") == {}


def test_redirects_are_not_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _net(
        monkeypatch,
        lambda r: httpx.Response(302, headers={"location": "https://evil.example/steal"}),
    )
    with pytest.raises(BrokerError):
        _client().get_history_page("orders", pacer=_pacer()[0], retries=0)
    assert [r.url.host for r in seen] == ["live.trading212.com"]


def test_client_only_issues_get_requests() -> None:
    source = inspect.getsource(t212)
    assert ".get(" in source
    for verb in (".post(", ".put(", ".patch(", ".delete(", ".request(", ".send("):
        assert verb not in source
    public = {n for n in dir(_client()) if not n.startswith("_")}
    assert public == {
        "BASE_URLS",
        "base_url",
        "environment",
        "get_account_summary",
        "get_history_page",
        "get_positions",
    }


# ------------------------------------------------------------------ error handling
@pytest.mark.parametrize(
    ("status", "code"), [(401, "auth"), (403, "scope"), (400, "http"), (404, "http")]
)
def test_non_retryable_statuses_fail_once(
    monkeypatch: pytest.MonkeyPatch, status: int, code: str
) -> None:
    seen = _net(monkeypatch, lambda r: httpx.Response(status, text="Authorization: leaked-detail"))
    with pytest.raises(BrokerError) as exc:
        _client().get_history_page("dividends", pacer=_pacer()[0])
    assert exc.value.code == code and len(seen) == 1
    assert KEY not in str(exc.value) and SECRET not in str(exc.value)
    assert "leaked-detail" not in str(exc.value)


def test_scope_error_names_the_history_scopes(monkeypatch: pytest.MonkeyPatch) -> None:
    _net(monkeypatch, lambda r: httpx.Response(403))
    with pytest.raises(BrokerError, match="history:orders"):
        _client().get_history_page("orders", pacer=_pacer()[0])


def test_429_is_retried_after_the_advertised_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"n": 0}

    def handler(r: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, headers={"x-ratelimit-reset": str(int(time.time()) + 12)})
        return httpx.Response(200, json={"items": [{"id": 1}], "nextPagePath": None})

    _net(monkeypatch, handler)
    pacer, sleeps = _pacer()
    items, _ = _client().get_history_page("orders", pacer=pacer)
    assert items == [{"id": 1}] and attempts["n"] == 2
    assert sleeps and 9 <= max(sleeps) <= 13


def test_429_with_unreasonable_reset_is_not_waited_out(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _net(
        monkeypatch,
        lambda r: httpx.Response(429, headers={"x-ratelimit-reset": str(int(time.time()) + 3000)}),
    )
    pacer, sleeps = _pacer()
    with pytest.raises(BrokerError) as exc:
        _client().get_history_page("orders", pacer=pacer)
    assert exc.value.code == "rate_limit" and len(seen) == 1 and not sleeps


def test_persistent_429_gives_up_after_bounded_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _net(monkeypatch, lambda r: httpx.Response(429))
    with pytest.raises(BrokerError) as exc:
        _client().get_history_page("orders", pacer=_pacer()[0])
    assert exc.value.code == "rate_limit" and len(seen) == 3


@pytest.mark.parametrize(
    ("status", "code"), [(500, "unavailable"), (503, "unavailable"), (408, "timeout")]
)
def test_transient_failures_are_retried_then_reported(
    monkeypatch: pytest.MonkeyPatch, status: int, code: str
) -> None:
    seen = _net(monkeypatch, lambda r: httpx.Response(status))
    with pytest.raises(BrokerError) as exc:
        _client().get_history_page("transactions", pacer=_pacer()[0])
    assert exc.value.code == code and len(seen) == 3


def test_transient_failure_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    n = {"i": 0}

    def handler(r: httpx.Request) -> httpx.Response:
        n["i"] += 1
        return httpx.Response(502) if n["i"] == 1 else httpx.Response(200, json={"items": []})

    _net(monkeypatch, handler)
    assert _client().get_history_page("orders", pacer=_pacer()[0]) == ([], None)


def test_timeouts_are_retried_and_then_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(f"timeout talking to {r.url} with {KEY}", request=r)

    seen = _net(monkeypatch, handler)
    with pytest.raises(BrokerError) as exc:
        _client().get_history_page("orders", pacer=_pacer()[0])
    assert exc.value.code == "network" and len(seen) == 3
    assert KEY not in str(exc.value) and "trading212" not in str(exc.value)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        "text",
        {"nextPagePath": None},
        {"items": "x"},
        {"items": [1, 2]},
        {"items": [], "nextPagePath": 5},
        {"items": [], "nextPagePath": {"a": 1}},
    ],
)
def test_malformed_page_shapes_are_rejected(monkeypatch: pytest.MonkeyPatch, payload: Any) -> None:
    _net(monkeypatch, lambda r: httpx.Response(200, json=payload))
    with pytest.raises(BrokerError) as exc:
        _client().get_history_page("orders", pacer=_pacer()[0])
    assert exc.value.code == "schema"


def test_non_json_body_is_a_safe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _net(monkeypatch, lambda r: httpx.Response(200, text="<html>nope</html>"))
    with pytest.raises(BrokerError):
        _client().get_history_page("orders", pacer=_pacer()[0], retries=0)


# ------------------------------------------------------------------ pacing
def test_pacer_spaces_requests_and_never_sleeps_when_enough_time_passed() -> None:
    now = {"t": 100.0}
    sleeps: list[float] = []
    pacer = HistoryPacer(3.0, clock=lambda: now["t"], sleep=sleeps.append)
    pacer.wait()  # first call: no delay
    now["t"] += 1.0
    pacer.wait()
    assert sleeps == [pytest.approx(2.0)]
    now["t"] += 10.0
    pacer.wait()
    assert sleeps == [pytest.approx(2.0)]
    pacer.wait(extra_s=7.0)
    assert sleeps[-1] == pytest.approx(7.0)


def test_default_pacing_respects_the_documented_20_per_minute() -> None:
    assert t212.HISTORY_MIN_INTERVAL_S * 20 >= 60
    assert t212.HISTORY_PAGE_LIMIT == 50


def test_source_has_no_credentials() -> None:
    text = Path(t212.__file__).read_text(encoding="utf-8")
    assert "SENTINEL" not in text

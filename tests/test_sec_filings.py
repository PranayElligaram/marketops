"""SEC EDGAR filing metadata: deterministic validation, provenance and API behaviour (offline)."""

from __future__ import annotations

from datetime import date
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes import filings as filings_route
from marketops.config import get_settings
from marketops.data import sec_filings
from marketops.data.sec_filings import SecClient, SecUnavailable, cik_for_ticker, parse_filings

TICKERS = {"0": {"cik_str": 1234567, "ticker": "EXMP", "title": "Example Corp"}}
TODAY = date(2026, 11, 7)


def _submissions(**recent: list[Any]) -> dict[str, Any]:
    base: dict[str, list[Any]] = {
        "accessionNumber": ["0001234567-26-000010", "0001234567-26-000009", "0001234567-26-000008"],
        "filingDate": ["2026-11-05", "2026-10-01", "2026-12-25"],
        "form": ["8-K", "4", "10-Q"],
        "primaryDocument": ["a8k.htm", "form4.xml", "q.htm"],
        "acceptanceDateTime": ["2026-11-05T21:30:11.000Z", "2026-10-01T10:00:00.000Z", ""],
        "primaryDocDescription": ["Current report", "Statement", "Quarterly"],
        "items": ["2.02,9.01", "", ""],
    }
    base.update(recent)
    return {"cik": "1234567", "name": "Example Corp", "filings": {"recent": base}}


def test_ticker_lookup_is_exact_and_validated() -> None:
    assert cik_for_ticker(TICKERS, "exmp") == (1234567, "Example Corp")
    for bad in ("NOPE", "../x", ""):
        with pytest.raises(KeyError):
            cik_for_ticker(TICKERS, bad)


def test_parse_keeps_only_valid_allowed_past_forms() -> None:
    name, rows = parse_filings(_submissions(), 1234567, today=TODAY)
    assert name == "Example Corp"
    # Form 4 is not an allowed form; the 10-Q dated after "today" is dropped, not repaired.
    assert [r.form for r in rows] == ["8-K"]
    row = rows[0]
    assert row.accession_number == "0001234567-26-000010"
    assert row.items == ["2.02", "9.01"]
    assert row.accepted_utc == "2026-11-05T21:30:11+00:00"
    assert row.url == "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000010/a8k.htm"


@pytest.mark.parametrize(
    "recent",
    [
        {"accessionNumber": ["../../etc/passwd"] * 3},
        {"primaryDocument": ["https://evil.example/x.htm", "b.htm", "c.htm"]},
        {"filingDate": ["not-a-date"] * 3},
    ],
)
def test_malformed_rows_are_dropped(recent: dict[str, list[Any]]) -> None:
    _, rows = parse_filings(_submissions(**recent), 1234567, today=TODAY)
    assert rows == []


def test_structural_errors_fail_closed() -> None:
    with pytest.raises(SecUnavailable):
        parse_filings({"cik": "999", "filings": {"recent": {}}}, 1234567, today=TODAY)
    with pytest.raises(SecUnavailable):
        parse_filings(_submissions(form=["8-K"]), 1234567, today=TODAY)
    with pytest.raises(SecUnavailable):
        parse_filings(["nonsense"], 1234567, today=TODAY)


def _client(monkeypatch: pytest.MonkeyPatch, payloads: dict[str, Any]) -> SecClient:
    client = SecClient("Test Suite test@example.com")
    monkeypatch.setattr(client, "_get_json", lambda url, ttl: payloads[url])
    return client


def test_client_adds_provenance_and_attribution(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(
        monkeypatch,
        {
            sec_filings.TICKERS_URL: TICKERS,
            sec_filings.SUBMISSIONS_URL.format(cik=1234567): _submissions(),
        },
    )
    out = client.recent_filings("EXMP")
    assert out["source_url"].startswith("https://data.sec.gov/submissions/CIK0001234567")
    assert out["retrieved_utc"].endswith("+00:00")
    assert "Securities and Exchange Commission" in out["attribution"]
    assert "no headline" in out["limitation"]
    assert out["filings"][0]["url"].startswith("https://www.sec.gov/Archives/")
    with pytest.raises(LookupError):
        client.recent_filings("NOPE")


def test_http_failures_become_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    class Boom:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> Boom:
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def get(self, url: str) -> httpx.Response:
            return httpx.Response(429, request=httpx.Request("GET", url))

    monkeypatch.setattr(sec_filings.httpx, "Client", Boom)
    monkeypatch.setattr(sec_filings, "MIN_INTERVAL_S", 0)
    with pytest.raises(SecUnavailable):
        SecClient("Test Suite test@example.com")._get_json("https://data.sec.gov/x", 0)


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("MARKETOPS_SEC_USER_AGENT", "Test Suite test@example.com")
    get_settings.cache_clear()
    filings_route._client.cache_clear()
    yield_client = TestClient(app)
    return yield_client


def test_endpoint_unconfigured_does_not_call_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARKETOPS_SEC_USER_AGENT", "")
    get_settings.cache_clear()
    monkeypatch.setattr(
        sec_filings.httpx, "Client", lambda **k: pytest.fail("network call without contact")
    )
    res = TestClient(app).get("/research/filings/EXMP")
    assert res.status_code == 503 and "MARKETOPS_SEC_USER_AGENT" in res.json()["detail"]


def test_endpoint_success_not_found_and_failure(
    api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {
        sec_filings.TICKERS_URL: TICKERS,
        sec_filings.SUBMISSIONS_URL.format(cik=1234567): _submissions(),
    }
    monkeypatch.setattr(SecClient, "_get_json", lambda self, url, ttl: payloads[url])
    ok = api.get("/research/filings/exmp")
    assert ok.status_code == 200 and ok.json()["filings"][0]["form"] == "8-K"
    assert api.get("/research/filings/NOPE").status_code == 404
    assert api.get("/research/filings/..%2Fx").status_code in (404, 422)

    def fail(self: SecClient, url: str, ttl: float) -> Any:
        raise SecUnavailable("down")

    monkeypatch.setattr(SecClient, "_get_json", fail)
    assert api.get("/research/filings/EXMP").status_code == 502

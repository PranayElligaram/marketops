"""Offline tests: broker access is read-only, with no account credentials or network."""

import httpx
import pytest
from fastapi.testclient import TestClient

from api.main import app
from marketops.broker.t212 import BrokerError, Trading212ReadOnlyClient
from marketops.config import Settings
from marketops.portfolio.analytics import EXAMPLE_POSITIONS, summarise_positions


def test_example_is_explicitly_synthetic() -> None:
    response = TestClient(app).get("/portfolio/example")
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "synthetic_example"
    assert body["holdings_value"] == 1000
    assert body["largest_weight_pct"] == 60
    assert body["unrealised_pnl"] == 20


def test_position_weights_use_wallet_values_not_quote_price() -> None:
    rows = [dict(x) for x in EXAMPLE_POSITIONS]
    rows[0] = dict(rows[0], currentPrice=1_000_000)
    overview = summarise_positions(rows, source="trading212")
    assert overview.holdings_value == 1000
    assert overview.holdings[0].weight_pct == 60


def test_mixed_currencies_rejected() -> None:
    rows = [dict(x) for x in EXAMPLE_POSITIONS]
    rows[1] = dict(rows[1], walletImpact=dict(rows[1]["walletImpact"], currency="EUR"))
    with pytest.raises(ValueError, match="mixed wallet currencies"):
        summarise_positions(rows, source="trading212")


def test_missing_wallet_values_rejected() -> None:
    rows = [dict(x) for x in EXAMPLE_POSITIONS]
    rows[1] = dict(rows[1], walletImpact={"currency": "GBP"})
    with pytest.raises(ValueError, match="Invalid monetary"):
        summarise_positions(rows, source="trading212")


def test_non_finite_values_rejected() -> None:
    rows = [dict(x) for x in EXAMPLE_POSITIONS]
    rows[0] = dict(rows[0], walletImpact=dict(rows[0]["walletImpact"], currentValue="nan"))
    with pytest.raises(ValueError, match="Non-finite"):
        summarise_positions(rows, source="trading212")


def test_broker_client_has_no_order_methods() -> None:
    client = Trading212ReadOnlyClient("key", "secret", "demo")
    assert not any("order" in name for name in dir(client))
    assert client.base_url == "https://demo.trading212.com/api/v0"


def test_broker_requires_both_credentials() -> None:
    with pytest.raises(BrokerError, match="not configured"):
        Trading212ReadOnlyClient("key", "", "demo")


def test_broker_uses_readonly_get_and_basic_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[dict[str, object]]:
            return EXAMPLE_POSITIONS

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            assert isinstance(kwargs["auth"], httpx.BasicAuth)

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def get(self, url: str) -> FakeResponse:
            calls.append(url)
            return FakeResponse()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    assert len(Trading212ReadOnlyClient("key", "secret", "demo").get_positions()) == 2
    assert calls == ["https://demo.trading212.com/api/v0/equity/positions"]


def test_keys_are_not_configured_by_default() -> None:
    settings = Settings(_env_file=None, t212_api_key="", t212_api_secret="")
    assert not settings.broker_configured

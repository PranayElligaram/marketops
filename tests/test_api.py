"""Contract tests run offline and without PostgreSQL."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api.main import app
from marketops.persistence import database


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def test_liveness_is_independent_of_database(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "marketops-api"}
    assert response.headers["X-Request-ID"]


def test_readiness_returns_503_when_db_down(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(database, "database_is_ready", lambda: False)
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "dependencies": {"postgres": "down"}}


def test_readiness_returns_200_when_db_up(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(database, "database_is_ready", lambda: True)
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "dependencies": {"postgres": "up"}}


def test_root_identifies_service(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["phase"] == "research"


def test_openapi_contains_only_implemented_endpoints(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert "/health/ready" in paths
    assert "/analysis/{symbol}" in paths
    # No order-placement or trading endpoints. "/history/orders" is a read-only listing of
    # already-imported historical orders (GET); "/history/sync" only imports data locally.
    assert not any(("order" in p and p != "/history/orders") or "trade" in p for p in paths)
    assert set(paths["/history/orders"]) == {"get"}
    writes = {
        (p, m)
        for p, ops in paths.items()
        for m in ops
        if m != "get" and p.startswith(("/history", "/portfolio"))
    }
    assert writes == {("/history/sync", "post")}
    assert "/investigate" not in paths

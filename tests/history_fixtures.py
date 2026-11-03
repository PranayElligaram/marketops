"""Synthetic Trading 212 history payloads (shapes from the OpenAPI schema; no real data)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from marketops.broker.t212 import BrokerError, HistoryKind, HistoryPacer

SUMMARY: dict[str, Any] = {
    "id": 424242,
    "currency": "GBP",
    "totalValue": 1000.0,
    "cash": {"availableToTrade": 100.0, "inPies": 0.0, "reservedForOrders": 0.0},
    "investments": {
        "currentValue": 900.0,
        "realizedProfitLoss": 12.5,
        "totalCost": 880.0,
        "unrealizedProfitLoss": 20.0,
    },
}


def order(
    n: int,
    side: str = "BUY",
    *,
    fill: bool = True,
    value: float = -100.0,
    pnl: float = 0.0,
    day: int = 10,
    ccy: str = "GBP",
    fill_type: str = "TRADE",
    taxes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "order": {
            "id": n,
            "ticker": "AAA_US_EQ",
            "side": side,
            "type": "MARKET",
            "status": "FILLED" if fill else "CANCELLED",
            "currency": "USD",
            "createdAt": f"2024-03-{day:02d}T09:00:00Z",
            "instrument": {"currency": "USD"},
        }
    }
    if fill:
        out["fill"] = {
            "id": n * 10,
            "filledAt": f"2024-03-{day:02d}T09:00:05Z",
            "price": 10.0,
            "quantity": 2.5,
            "type": fill_type,
            "tradingMethod": "TOTV",
            "walletImpact": {
                "currency": ccy,
                "fxRate": 1.25,
                "netValue": value,
                "realisedProfitLoss": pnl,
                "taxes": taxes or [],
            },
        }
    return out


def tx(ref: str, kind: str, amount: float, *, day: int = 5, ccy: str = "GBP") -> dict[str, Any]:
    return {
        "reference": ref,
        "type": kind,
        "amount": amount,
        "currency": ccy,
        "dateTime": f"2024-02-{day:02d}T12:00:00Z",
    }


def div(
    ref: str, amount: float = 3.2, *, kind: str = "ORDINARY", day: int = 15, ccy: str = "GBP"
) -> dict[str, Any]:
    return {
        "reference": ref,
        "ticker": "AAA_US_EQ",
        "type": kind,
        "amount": amount,
        "amountInEuro": amount * 1.1,
        "currency": ccy,
        "grossAmountPerShare": 0.5,
        "quantity": 8,
        "tickerCurrency": "USD",
        "instrument": {"currency": "USD"},
        "paidOn": f"2024-04-{day:02d}T00:00:00Z",
    }


def sqlite_engine(tmp_path: Any, name: str = "h.db") -> Engine:
    return create_engine(f"sqlite:///{tmp_path / name}")


def fast_pacer() -> HistoryPacer:
    return HistoryPacer(0, sleep=lambda s: None)


class FakeSource:
    """In-memory broker: `pages[kind]` is a list of pages; nextPagePath mimics the real shape."""

    def __init__(
        self,
        pages: dict[HistoryKind, list[list[dict[str, Any]]]],
        summary: dict[str, Any] | None = None,
        environment: str = "demo",
    ) -> None:
        self.pages = pages
        self.summary = SUMMARY if summary is None else summary
        self.environment = environment
        self.fail_at: dict[tuple[str, int], BrokerError] = {}
        self.next_override: dict[tuple[str, int], str | None] = {}
        self.calls: list[tuple[str, int]] = []

    def get_account_summary(self) -> dict[str, Any]:
        return self.summary

    def get_history_page(
        self, kind: HistoryKind, next_page_path: str | None = None, pacer: Any = None
    ) -> tuple[list[dict[str, Any]], str | None]:
        idx = 0 if next_page_path is None else int(next_page_path.rsplit("cursor=", 1)[1])
        self.calls.append((kind, idx))
        if (kind, idx) in self.fail_at:
            raise self.fail_at[(kind, idx)]
        pages = self.pages.get(kind, [[]])
        nxt: str | None = (
            f"/api/v0/equity/history/{kind}?limit=50&cursor={idx + 1}"
            if idx + 1 < len(pages)
            else None
        )
        nxt = self.next_override.get((kind, idx), nxt)
        return pages[idx], nxt


def standard_pages() -> dict[HistoryKind, list[list[dict[str, Any]]]]:
    return {
        "orders": [
            [order(5, "SELL", value=120.0, pnl=20.0, day=20), order(4, day=19)],
            [order(3, day=18, taxes=[{"name": "STAMP_DUTY", "currency": "GBP", "quantity": 0.5}])],
            [order(2, "BUY", fill=False, day=10)],
        ],
        "transactions": [
            [tx("T4", "WITHDRAW", -50.0, day=20), tx("T3", "FEE", -1.0, day=19)],
            [tx("T2", "DEPOSIT", 500.0, day=10), tx("T1", "DEPOSIT", 200.0, day=3)],
        ],
        "dividends": [[div("D2", 4.0, day=20)], [div("D1", 3.2, day=15)]],
    }


TOKEN = "tok-0123456789abcdef-SENTINEL"
HDR = {"X-MarketOps-Token": TOKEN}
KEY, SECRET = "k-SENTINEL-key", "s-SENTINEL-secret"


class Env:
    """FastAPI client + SQLite engine + scripted broker, created by the `env` fixture."""

    def __init__(self, engine: Engine, client: Any) -> None:
        self.engine, self.client = engine, client
        self.source = FakeSource(standard_pages())
        self.constructed = 0

    def sync(self, full: bool = False) -> Any:
        return self.client.post("/history/sync", json={"full": full}, headers=HDR)

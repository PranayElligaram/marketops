"""Independent synthetic checks for the fixed-rule medium-term research backtest."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes import decision
from api.routes.research import Bar
from marketops.decision.strategy import evaluate_position_strategy


def prices(n: int = 330, *, gap_down: bool = False) -> pd.DataFrame:
    idx = np.arange(n, dtype=float)
    close = 100.0 + idx * 0.25
    if n > 220:
        close[220:] += 12.0
    if gap_down and n > 226:
        close[226:] -= 40.0
    volume = np.full(n, 1000.0)
    if n > 220:
        volume[220] = 3000.0
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=n),
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "adj_close": close,
            "volume": volume,
        }
    )


def test_fixed_entry_exit_and_next_session_execution() -> None:
    frame = prices()
    report = evaluate_position_strategy(frame, "AAA")
    assert report["status"] == "available"
    signals = report["signals"]
    assert signals[0]["action"] == "enter"
    assert signals[0]["date"] == str(frame.iloc[220]["date"].date())
    assert signals[0]["execution"] == str(frame.iloc[221]["date"].date())
    assert any(x["action"] == "exit" and "60-session" in x["reason"] for x in signals)
    # The equity baseline is at the preceding signal close, before the next-open fill.
    assert report["window_start"] == signals[0]["date"]
    assert report["window_end"] == str(frame.iloc[-1]["date"].date())
    assert report["strategy"] is not None and report["buy_and_hold"] is not None
    assert report["trades"][0]["date"] == signals[0]["execution"]


def test_close_breakdown_exits_at_next_open_not_intraday_stop() -> None:
    report = evaluate_position_strategy(prices(gap_down=True), "AAA")
    assert report["status"] == "available"
    exits = [s for s in report["signals"] if s["action"] == "exit"]
    assert exits and "breakdown" in exits[0]["reason"]
    assert exits[0]["execution"] > exits[0]["date"]


def test_no_volume_confirmation_gives_no_action() -> None:
    frame = prices()
    frame.loc[:, "volume"] = 1000.0
    report = evaluate_position_strategy(frame, "AAA")
    assert report["status"] == "no_action"
    assert report["strategy"] is None
    assert report["buy_and_hold"] is None


def test_insufficient_history_does_not_invent_performance() -> None:
    result = evaluate_position_strategy(prices(n=200), "AAA")
    assert result["status"] == "unavailable"
    assert result["strategy"] is None


def test_future_prices_do_not_change_past_signals() -> None:
    earlier = prices()
    modified = earlier.copy()
    modified.loc[310:, "close"] *= 0.8
    modified.loc[310:, "open"] = modified.loc[310:, "close"] - 0.1
    modified.loc[310:, "high"] = modified.loc[310:, "close"] + 0.5
    modified.loc[310:, "low"] = modified.loc[310:, "close"] - 0.5
    modified.loc[310:, "adj_close"] = modified.loc[310:, "close"]
    before = evaluate_position_strategy(earlier, "AAA")
    after = evaluate_position_strategy(modified, "AAA")
    assert [x for x in before["signals"] if x["date"] < "2025-03-01"] == [
        x for x in after["signals"] if x["date"] < "2025-03-01"
    ]


@pytest.mark.parametrize("field", ["open", "high", "low", "close", "adj_close"])
def test_rejects_nonpositive_prices(field: str) -> None:
    frame = prices()
    frame.loc[5, field] = 0.0
    with pytest.raises(ValueError):
        evaluate_position_strategy(frame, "AAA")


def test_rejects_nan_and_unsorted_bars() -> None:
    frame = prices()
    frame.loc[5, "volume"] = np.nan
    with pytest.raises(ValueError):
        evaluate_position_strategy(frame, "AAA")
    frame = prices()
    frame.loc[2, "date"] = frame.loc[1, "date"]
    with pytest.raises(ValueError):
        evaluate_position_strategy(frame, "AAA")


def test_rejects_invalid_cost_and_symbol() -> None:
    with pytest.raises(ValueError):
        evaluate_position_strategy(prices(), "AAA", cost_bps=-1)
    with pytest.raises(ValueError):
        evaluate_position_strategy(prices(), "../secret")


def test_api_historical_strategy_is_only_synthetic_in_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = prices()
    bars = []
    for row in frame.to_dict("records"):
        bars.append(
            Bar(
                date=str(row["date"].date()),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                adj_close=float(row["adj_close"]),
                volume=float(row["volume"]),
            )
        )
    monkeypatch.setattr(
        decision,
        "market_history",
        lambda symbol, days: SimpleNamespace(
            symbol=symbol.upper(),
            rows=bars,
            provider="synthetic_test",
            price_convention="synthetic",
            terms="Not real market data",
        ),
    )
    with TestClient(app) as client:
        response = client.get("/research/position-strategy/AAA")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "available"
    assert body["provider"] == "synthetic_test"
    assert body["window_start"] == body["signals"][0]["date"]
    assert body["strategy"]["cagr"] is not None

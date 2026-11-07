"""Synthetic tests for historical technical review and hypothetical risk budgeting."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes import decision
from marketops.decision.technical import (
    DailyBar,
    screen_daily_bars,
    size_risk_scenario,
)


def history(
    *,
    n: int = 220,
    finish: date = date(2026, 11, 6),
    last_close: float | None = None,
    last_volume: float = 3000.0,
) -> list[DailyBar]:
    days = [finish - timedelta(days=n - i - 1) for i in range(n)]
    out = [
        DailyBar(
            day=day,
            close=100.0 + i * 0.25,
            high=101.0 + i * 0.25,
            low=99.0 + i * 0.25,
            volume=1000.0,
        )
        for i, day in enumerate(days)
    ]
    if last_close is not None:
        last = out[-1]
        out[-1] = DailyBar(
            day=last.day,
            close=last_close,
            high=max(last_close + 1, last.high),
            low=min(last_close - 1, last.low),
            volume=last_volume,
        )
    return out


def test_breakout_uses_only_prior_days() -> None:
    bars = history(last_close=180)
    result = screen_daily_bars(bars, as_of=date(2026, 11, 7))
    assert result.state == "breakout_review"
    assert result.prior_20_high is not None and result.prior_20_high < 180
    assert result.volume_ratio_20 == pytest.approx(3.0)
    assert result.atr_14 is not None and result.atr_14 > 0


def test_trend_without_volume_confirmation_is_watch() -> None:
    result = screen_daily_bars(history(last_close=160, last_volume=1000), as_of=date(2026, 11, 7))
    assert result.state == "trend_watch"


def test_below_sma_is_exit_review_not_order() -> None:
    result = screen_daily_bars(history(last_close=80), as_of=date(2026, 11, 7))
    assert result.state == "exit_review"
    assert "not an instruction" in " ".join(result.rationale).lower()


def test_stale_and_insufficient_fail_closed() -> None:
    stale = screen_daily_bars(history(), as_of=date(2026, 11, 30))
    assert stale.state == "unavailable"
    assert stale.close is None
    short = screen_daily_bars(history(n=199), as_of=date(2026, 11, 7))
    assert short.state == "unavailable"
    assert short.sma_200 is None


def test_empty_history_unavailable() -> None:
    result = screen_daily_bars([], as_of=date(2026, 11, 7))
    assert result.state == "unavailable"
    assert result.as_of is None


def test_prefix_independence() -> None:
    bars = history(n=225)
    prefix = screen_daily_bars(bars[:220])
    extended = history(n=220, finish=bars[219].day)
    # Historical calendar dates match; future rows cannot alter the earlier screen.
    assert screen_daily_bars(extended).close == prefix.close


@pytest.mark.parametrize(
    "bar",
    [
        DailyBar(date(2026, 11, 7), 0, 1, 0.5, 100),
        DailyBar(date(2026, 11, 7), 2, 1, 1.5, 100),
        DailyBar(date(2026, 11, 7), 3, 1, 2, -1),
        DailyBar(date(2026, 11, 7), float("nan"), 1, 1, 100),
        DailyBar(date(2026, 11, 7), 3, 1, 2, float("inf")),
    ],
)
def test_invalid_ohlcv_rejected(bar: DailyBar) -> None:
    with pytest.raises(ValueError):
        screen_daily_bars([bar])


def test_unsorted_and_duplicate_dates_rejected() -> None:
    b = history(n=2)
    with pytest.raises(ValueError):
        screen_daily_bars([b[1], b[0]])
    with pytest.raises(ValueError):
        screen_daily_bars([b[0], b[0]])


def test_scenario_risk_budget_binds() -> None:
    r = size_risk_scenario(
        10000.0, 0.0, loss_budget_pct=1.0, position_cap_pct=20.0, stop_distance_pct=10.0
    )
    assert r.maximum_additional_notional_gbp == pytest.approx(1000)
    assert r.risk_budget_gbp == pytest.approx(100)
    assert r.binding_limit == "risk budget"


def test_scenario_existing_exposure_binds() -> None:
    r = size_risk_scenario(10000, 950, position_cap_pct=10)
    assert r.maximum_additional_notional_gbp == pytest.approx(50)
    assert r.binding_limit == "concentration cap"
    assert "not guaranteed" in r.limitation


def test_zero_capacity_is_not_a_trade_signal() -> None:
    r = size_risk_scenario(10000, 2500, position_cap_pct=10)
    assert r.maximum_additional_notional_gbp == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"portfolio_value_gbp": 0, "existing_exposure_gbp": 0},
        {"portfolio_value_gbp": 1000, "existing_exposure_gbp": 2000},
        {"portfolio_value_gbp": 1000, "existing_exposure_gbp": -1},
        {"portfolio_value_gbp": float("nan"), "existing_exposure_gbp": 0},
        {"portfolio_value_gbp": 1000, "existing_exposure_gbp": 0, "loss_budget_pct": 0},
        {"portfolio_value_gbp": 1000, "existing_exposure_gbp": 0, "position_cap_pct": 30},
        {"portfolio_value_gbp": 1000, "existing_exposure_gbp": 0, "stop_distance_pct": 0},
    ],
)
def test_scenario_rejects_invalid_inputs(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        size_risk_scenario(**kwargs)


def test_api_research_screen_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = history(last_close=180)
    values = [
        SimpleNamespace(
            date=x.day.isoformat(), high=x.high, low=x.low, close=x.close, volume=x.volume
        )
        for x in entries
    ]
    monkeypatch.setattr(
        decision,
        "market_history",
        lambda symbol, days: SimpleNamespace(
            rows=values,
            symbol=symbol.upper(),
            provider="synthetic_fixture",
            price_convention="synthetic",
            terms="Test only",
        ),
    )
    with TestClient(app) as c:
        response = c.get("/research/technical/AAA")
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "AAA"
    assert body["state"] == "breakout_review"
    assert body["provider"] == "synthetic_fixture"
    assert body["limitation"].startswith("Historical conditions")
    assert "forecast" not in body


def test_api_invalid_cache_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(
        symbol="AAA",
        provider="synthetic",
        price_convention="synthetic",
        terms="Test",
        rows=[SimpleNamespace(date="2026-11-06", high=1, low=2, close=1, volume=1)],
    )
    monkeypatch.setattr(decision, "market_history", lambda symbol, days: fake)
    with TestClient(app) as c:
        result = c.get("/research/technical/AAA")
    assert result.status_code == 502
    assert "invalid" in result.json()["detail"]

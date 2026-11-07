"""Deterministic position-trading research, not an execution or prediction engine.

Only historical daily OHLCV values are used. This produces research conditions rather
than buy/sell advice or forecasts. Insufficient or stale input fails closed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

ScreenState = Literal["breakout_review", "trend_watch", "exit_review", "no_action", "unavailable"]


@dataclass(frozen=True)
class DailyBar:
    day: date
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class TechnicalScreen:
    as_of: date | None
    state: ScreenState
    close: float | None
    sma_50: float | None
    sma_200: float | None
    prior_20_high: float | None
    atr_14: float | None
    volume_ratio_20: float | None
    rationale: tuple[str, ...]
    limitation: str
    horizon: str = "2–12 weeks"


@dataclass(frozen=True)
class RiskScenario:
    portfolio_value_gbp: float
    existing_exposure_gbp: float
    risk_budget_gbp: float
    maximum_additional_notional_gbp: float
    binding_limit: str
    limitation: str


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def screen_daily_bars(bars: Sequence[DailyBar], *, as_of: date | None = None) -> TechnicalScreen:
    """Evaluate the latest bar using historical data only.

    The prior-20-day high and volume reference explicitly exclude the latest bar.
    200 sessions are required for the long trend. as_of checks cache freshness.
    """
    previous: date | None = None
    for bar in bars:
        if type(bar.day) is not date or (previous is not None and bar.day <= previous):
            raise ValueError("daily bars must have strictly increasing dates")
        vals = (bar.high, bar.low, bar.close, bar.volume)
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in vals):
            raise ValueError("daily OHLCV must be finite numbers")
        if (
            bar.low <= 0
            or bar.close <= 0
            or bar.high < max(bar.close, bar.low)
            or bar.close < bar.low
        ):
            raise ValueError("daily OHLCV must contain valid positive prices")
        if bar.volume < 0:
            raise ValueError("volume must be nonnegative")
        previous = bar.day
    if not bars:
        return TechnicalScreen(
            None,
            "unavailable",
            None,
            None,
            None,
            None,
            None,
            None,
            ("No historical prices available.",),
            "Historical research only; no forecast or trade instruction.",
        )
    last = bars[-1]
    if as_of is not None and (as_of < last.day or (as_of - last.day).days > 7):
        return TechnicalScreen(
            last.day,
            "unavailable",
            None,
            None,
            None,
            None,
            None,
            None,
            ("Market data is dated in the future or more than seven days old.",),
            "Refresh price history before evaluating conditions.",
        )
    if len(bars) < 200:
        return TechnicalScreen(
            last.day,
            "unavailable",
            last.close,
            None,
            None,
            None,
            None,
            None,
            (f"Need 200 price sessions; only {len(bars)} available.",),
            "No technical assessment with insufficient history.",
        )

    closes = [bar.close for bar in bars]
    sma50 = _mean(closes[-50:])
    sma200 = _mean(closes[-200:])
    prior_high = max(bar.high for bar in bars[-21:-1])
    prior_volume = _mean([bar.volume for bar in bars[-21:-1]])
    volume_ratio = last.volume / prior_volume if prior_volume > 0 else None
    true_ranges = []
    for i in range(len(bars) - 14, len(bars)):
        bar, previous_close = bars[i], bars[i - 1].close
        true_ranges.append(
            max(
                bar.high - bar.low,
                abs(bar.high - previous_close),
                abs(bar.low - previous_close),
            )
        )
    atr = _mean(true_ranges)
    trending = last.close > sma50 > sma200
    breakout = last.close > prior_high and trending
    volume_supported = volume_ratio is not None and volume_ratio >= 1.2
    if breakout and volume_supported:
        state: ScreenState = "breakout_review"
        rationale = (
            "Close exceeded the preceding 20-session high.",
            "Price is above the 50- and 200-session moving averages.",
            "Volume exceeded 1.2x the preceding 20-session average.",
        )
    elif trending:
        state = "trend_watch"
        rationale = (
            "Close is above the 50-session average, which is above the 200-session average.",
            "A breakout with volume confirmation is not established.",
        )
    elif last.close < sma50:
        state = "exit_review"
        rationale = (
            "Close is below the 50-session average; review any existing research thesis.",
            "This is not an instruction to sell.",
        )
    else:
        state = "no_action"
        rationale = ("The predefined trend and breakout conditions are not met.",)
    return TechnicalScreen(
        last.day,
        state,
        last.close,
        sma50,
        sma200,
        prior_high,
        atr,
        volume_ratio,
        rationale,
        "Historical conditions only; no prediction, target price or execution.",
    )


def size_risk_scenario(
    portfolio_value_gbp: float,
    existing_exposure_gbp: float,
    *,
    loss_budget_pct: float = 1.0,
    position_cap_pct: float = 10.0,
    stop_distance_pct: float = 8.0,
) -> RiskScenario:
    """Hypothetical maximum capital constrained by loss budget and concentration.

    No brokerage data is accessed, no price/FX conversion or share quantity is claimed,
    and actual available cash is not checked.
    """
    values = (
        portfolio_value_gbp,
        existing_exposure_gbp,
        loss_budget_pct,
        position_cap_pct,
        stop_distance_pct,
    )
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("scenario inputs must be finite numbers")
    if portfolio_value_gbp <= 0 or not 0 <= existing_exposure_gbp <= portfolio_value_gbp:
        raise ValueError("invalid portfolio value or existing exposure")
    if not 0 < loss_budget_pct <= 5 or not 0 < position_cap_pct <= 25:
        raise ValueError("risk budget and position cap outside supported research limits")
    if not 0 < stop_distance_pct <= 50:
        raise ValueError("stop distance outside supported research limits")
    risk_budget = portfolio_value_gbp * loss_budget_pct / 100
    by_risk = risk_budget / (stop_distance_pct / 100)
    by_cap = max(0.0, portfolio_value_gbp * position_cap_pct / 100 - existing_exposure_gbp)
    amount = min(by_risk, by_cap)
    binding = "risk budget" if by_risk < by_cap else "concentration cap"
    return RiskScenario(
        portfolio_value_gbp,
        existing_exposure_gbp,
        risk_budget,
        amount,
        binding,
        "Illustrative only. Stops are not guaranteed; fees, slippage, FX, "
        "cash available and correlated positions are not modelled.",
    )

"""Historical 2–12-week position-strategy research, not an investment signal.

A fixed long-only rule uses only data available at the signal close. The existing
backtester executes changes at the next session's open and charges one-way costs.
No optimisation, parameter search, broker access or claim of predictive skill.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd

from marketops.backtest.engine import BacktestResult, backtest

MIN_BARS = 201
MIN_HOLD_SESSIONS = 10
MAX_HOLD_SESSIONS = 60
ENTRY_VOLUME_MULTIPLE = 1.2
CLOSE_BREAKDOWN_PCT = 8.0
COST_BPS_ONE_WAY = 10.0


def _verified_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close", "adj_close", "volume"}
    if not required <= set(frame.columns):
        raise ValueError("Missing historical OHLCV columns")
    data = frame.loc[:, sorted(required)].copy()
    try:
        data["date"] = pd.to_datetime(data["date"], errors="raise")
        values = data[["open", "high", "low", "close", "adj_close", "volume"]].astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid historical OHLCV values") from exc
    if data["date"].isna().any() or not data["date"].is_monotonic_increasing:
        raise ValueError("Dates must be sorted ascending")
    if data["date"].duplicated().any():
        raise ValueError("Duplicate price date")
    if not bool(np.isfinite(values.to_numpy()).all()):
        raise ValueError("Non-finite price or volume")
    if bool((values[["open", "high", "low", "close", "adj_close"]] <= 0).any().any()):
        raise ValueError("Prices must be positive")
    if bool((values["volume"] < 0).any()):
        raise ValueError("Negative volume")
    if bool((values["high"] < values[["open", "close", "low"]].max(axis=1)).any()):
        raise ValueError("High below another bar price")
    if bool((values["low"] > values[["open", "close"]].min(axis=1)).any()):
        raise ValueError("Low above another bar price")
    return data.reset_index(drop=True)


def _metric_values(result: BacktestResult) -> dict[str, float | None]:
    return {
        name: float(value) if math.isfinite(float(value)) else None
        for name, value in result.metrics.items()
    }


def evaluate_position_strategy(
    frame: pd.DataFrame, symbol: str, *, cost_bps: float = COST_BPS_ONE_WAY
) -> dict[str, Any]:
    """Evaluate fixed historical rules with no parameter optimisation or future bars.

    Entry: close above preceding 20-bar high, close > SMA50 > SMA200, and
    current volume >= 1.2x preceding 20-bar average.
    Exit: close <= entry-signal-close * 0.92 (not a resting stop order);
    after 10 held sessions a close below SMA50; or after 60 held sessions.
    Signals are observed at the close and filled by backtest() at NEXT open.
    The baseline buys on the same first execution day and holds to the same end.
    """
    if not 0 <= cost_bps <= 100:
        raise ValueError("Cost must be 0–100 bps one-way")
    symbol = symbol.strip().upper()
    if not symbol or len(symbol) > 20 or not all(
        c.isalnum() or c in ".-" for c in symbol
    ):
        raise ValueError("Invalid research symbol")
    data = _verified_frame(frame)
    assumptions = (
        "Fixed rules, long-only, no leverage, 10–60-session holding window except "
        "close-based drawdown exit; no tuning or demonstrated edge.",
        "Signals at close, hypothetical fills at next open using adjusted prices. "
        "Breakdown checks are not broker stop orders and may gap.",
        f"One-way cost {cost_bps:g} bps; cash earns 0%; no explicit taxes, FX trading "
        "costs, funding costs or market-impact model.",
        "Comparison is historical, survivorship-biased and limited to the cached stock universe.",
    )
    base: dict[str, Any] = {
        "symbol": symbol,
        "status": "unavailable",
        "horizon": "2–12 weeks",
        "assumptions": list(assumptions),
        "signals": [],
        "strategy": None,
        "buy_and_hold": None,
        "trades": [],
        "window_start": None,
        "window_end": None,
        "reason": "",
    }
    if len(data) < MIN_BARS:
        base["reason"] = f"Need {MIN_BARS} valid sessions; only {len(data)} available."
        return base
    closes = data["close"].to_numpy()
    highs = data["high"].to_numpy()
    volumes = data["volume"].to_numpy()
    dates = pd.DatetimeIndex(data["date"])
    holdings = False
    entry_execution_index = -1
    entry_signal_close = 0.0
    signal_dates: list[pd.Timestamp] = []
    weights: list[float] = []
    for i in range(200, len(data) - 1):
        sma50 = float(np.mean(closes[i - 49 : i + 1]))
        sma200 = float(np.mean(closes[i - 199 : i + 1]))
        prior_high = float(np.max(highs[i - 20 : i]))
        prior_volume = float(np.mean(volumes[i - 20 : i]))
        action: Literal["enter", "exit"] | None = None
        reason = ""
        if not holdings:
            if (
                closes[i] > prior_high
                and closes[i] > sma50 > sma200
                and prior_volume > 0
                and volumes[i] >= ENTRY_VOLUME_MULTIPLE * prior_volume
            ):
                action = "enter"
                reason = "20-session breakout, trend and prior-volume confirmation"
                holdings = True
                entry_execution_index = i + 1
                entry_signal_close = float(closes[i])
        else:
            held = i - entry_execution_index + 1
            if closes[i] <= entry_signal_close * (1 - CLOSE_BREAKDOWN_PCT / 100):
                action = "exit"
                reason = "Closing-price breakdown vs entry signal; next-open exit only"
            elif held >= MAX_HOLD_SESSIONS:
                action = "exit"
                reason = "Maximum 60-session holding period"
            elif held >= MIN_HOLD_SESSIONS and closes[i] < sma50:
                action = "exit"
                reason = "Close below SMA50 after minimum 10 sessions"
            if action == "exit":
                holdings = False
        if action is not None:
            signal_dates.append(dates[i])
            weights.append(1.0 if action == "enter" else 0.0)
            base["signals"].append(
                {
                    "date": str(dates[i].date()),
                    "execution": str(dates[i + 1].date()),
                    "action": action,
                    "reason": reason,
                }
            )
    if not signal_dates:
        base["status"] = "no_action"
        base["reason"] = "No qualifying entry in the available historical window."
        return base

    targets = pd.DataFrame({symbol: weights}, index=pd.DatetimeIndex(signal_dates))
    first_signal = signal_dates[0]
    baseline = pd.DataFrame({symbol: [1.0]}, index=pd.DatetimeIndex([first_signal]))
    prices = {symbol: data}
    actual = backtest(prices, targets, cost_bps=cost_bps, end=dates[-1])
    passive = backtest(prices, baseline, cost_bps=cost_bps, end=dates[-1])
    if not actual.equity.index.equals(passive.equity.index):
        raise ValueError("Strategy and baseline valuation dates differ")
    base["status"] = "available"
    base["reason"] = "Historical backtest, not a validated live trading strategy."
    base["strategy"] = _metric_values(actual)
    base["buy_and_hold"] = _metric_values(passive)
    base["window_start"] = str(actual.equity.index[0].date())
    base["window_end"] = str(actual.equity.index[-1].date())
    base["trades"] = [
        {
            "date": str(row.date.date()),
            "symbol": row.symbol,
            "delta_weight": float(row.delta_weight),
            "cost_fraction": float(row.cost),
        }
        for row in actual.trades.itertuples(index=False)
    ]
    return base

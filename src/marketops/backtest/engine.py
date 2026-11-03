"""Long-only research backtester with explicit timing and costs.

Timing: a target-weight vector is computed from information available at the CLOSE of signal
date t. It is executed at the OPEN of the next session (t+1) at the adjusted open price. A
signal can never fill at or before the bar it was computed on.

Prices: adjusted prices (split and dividend adjusted, i.e. dividends are implicitly reinvested
on the ex-date). adj_open = open * adj_close / close. Cash earns 0%.

Costs: `cost_bps` is charged on traded notional (|weight change| x NAV) at every execution,
a one-way proxy for spread/slippage; Trading 212 charges no commission on stocks but FX
and spread costs exist. Default 10 bps is a deliberately conservative assumption, not a quote.

Between rebalances positions drift (no daily rebalancing). NAV is marked at adjusted close.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    equity: pd.Series  # NAV by date (starts at 1.0 at first execution)
    trades: pd.DataFrame  # date, symbol, delta_weight, cost
    turnover: float  # total |traded notional| / average NAV, annualised
    total_cost: float  # fraction of initial NAV
    metrics: dict[str, float]


def rebalance_dates(dates: pd.DatetimeIndex, every: int) -> pd.DatetimeIndex:
    return dates[::every]


def performance(equity: pd.Series) -> dict[str, float]:
    r = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else 0.0
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)
    mean = float(r.mean() * TRADING_DAYS)
    dd = equity / equity.cummax() - 1
    return {
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": float(mean / vol) if vol > 0 else float("nan"),
        "max_drawdown": float(dd.min()),
        "years": float(years),
    }


def backtest(
    prices: dict[str, pd.DataFrame],
    targets: pd.DataFrame,
    cost_bps: float = 10.0,
    end: pd.Timestamp | None = None,
) -> BacktestResult:
    """Run the simulation.

    prices: symbol -> frame with date, open, close, adj_close.
    targets: index = signal date (close), columns = symbols, values = target weights (>=0,
        sum <= 1; remainder is cash). Each row is executed at the next session's open.
    end: last date included. The calendar is cut here, so NAV is never marked on later prices
        and a signal whose next-session execution falls after `end` is dropped, not shifted.
    """
    if targets.isna().any().any():
        raise ValueError("Targets contain missing weights")
    if (targets.to_numpy() < -1e-12).any() or (targets.sum(axis=1) > 1 + 1e-9).any():
        raise ValueError("Targets must be long-only with total weight <= 1")
    syms = list(targets.columns)
    px = {
        s: prices[s]
        .set_index("date")
        .assign(adj_open=lambda d: d["open"] * d["adj_close"] / d["close"])
        for s in syms
    }
    cal = px[syms[0]].index
    for s in syms[1:]:
        cal = cal.intersection(px[s].index)
    if end is not None:
        cal = cal[cal <= end]
    adj_close = pd.DataFrame({s: px[s]["adj_close"].reindex(cal) for s in syms})
    adj_open = pd.DataFrame({s: px[s]["adj_open"].reindex(cal) for s in syms})
    if len(cal) == 0:
        raise ValueError("Symbols share no common trading dates")
    if adj_close.isna().any().any() or adj_open.isna().any().any() or (adj_open <= 0).any().any():
        raise ValueError("Missing or non-positive prices on shared trading dates")

    # Map each signal date to its execution date: the next session in the calendar.
    pos = cal.searchsorted(targets.index, side="right")  # first date strictly after signal
    exec_idx = {int(p): targets.iloc[i] for i, p in enumerate(pos) if p < len(cal)}
    if not exec_idx:
        raise ValueError("No executable signals inside price history")
    first = min(exec_idx)

    nav_cash = 1.0
    shares = pd.Series(0.0, index=syms)
    equity: dict[pd.Timestamp, float] = {}
    if first > 0:
        equity[cal[first - 1]] = 1.0  # capital committed before the first open
    trades = []
    traded_notional = 0.0
    total_cost = 0.0
    for i in range(first, len(cal)):
        d = cal[i]
        if i in exec_idx:
            o = adj_open.iloc[i]
            nav = nav_cash + float((shares * o).sum())
            cur_w = (shares * o) / nav
            tgt = exec_idx[i].astype(float)
            delta = tgt - cur_w
            notional = float(delta.abs().sum() * nav)
            cost = notional * cost_bps / 1e4
            nav_after = nav - cost
            shares = tgt * nav_after / o
            nav_cash = nav_after - float((shares * o).sum())
            traded_notional += notional
            total_cost += cost
            for s in syms:
                if abs(delta[s]) > 1e-12:
                    trades.append(
                        {
                            "date": d,
                            "symbol": s,
                            "delta_weight": float(delta[s]),
                            "cost": float(abs(delta[s]) * nav * cost_bps / 1e4),
                        }
                    )
        equity[d] = nav_cash + float((shares * adj_close.iloc[i]).sum())
    eq = pd.Series(equity, name="equity")
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    turnover = traded_notional / float(eq.mean()) / years
    m = performance(eq)
    m["annual_turnover"] = float(turnover)
    m["total_cost_frac"] = float(total_cost)
    return BacktestResult(
        eq,
        pd.DataFrame(trades, columns=["date", "symbol", "delta_weight", "cost"]),
        float(turnover),
        float(total_cost),
        m,
    )

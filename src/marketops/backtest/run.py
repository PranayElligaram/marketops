"""Strategy comparison for a trained model's saved predictions.

Strategies (all long-only, equal-weight, 5-session rebalance, same costs):
- model: hold symbols whose predicted P(up) >= the development-chosen threshold, else cash
- momentum_126d: hold symbols with positive 126-session return (rule fixed a priori, untuned)
- equal_weight: always hold the whole universe
- spy_buy_hold: benchmark, single initial purchase, never rebalanced
Dev = out-of-fold predictions (2010-2021); the dev curve is cut at dev_end so it is never
marked on test-period prices. Test = final model on held-out 2022+.
No parameter here was tuned on the test period.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from marketops.backtest.engine import BacktestResult, backtest, rebalance_dates
from marketops.data.store import BENCHMARK, load_symbol
from marketops.features.build import HORIZON


def _weights(signal: pd.DataFrame) -> pd.DataFrame:
    """Boolean date x symbol -> equal weight among selected (rest cash)."""
    n = signal.sum(axis=1).replace(0, float("nan"))
    return signal.astype(float).div(n, axis=0).fillna(0.0)


def check_period(
    dates: pd.DatetimeIndex, start: pd.Timestamp | None, end: pd.Timestamp | None, what: str
) -> None:
    """Refuse dates outside the split's boundaries instead of silently evaluating them."""
    if len(dates) == 0:
        raise ValueError(f"No {what} in this split")
    if start is not None and dates.min() < start:
        raise ValueError(f"{what} start {dates.min().date()} before period start {start.date()}")
    if end is not None and dates.max() > end:
        raise ValueError(f"{what} end {dates.max().date()} after period end {end.date()}")


def run_split(
    prices: dict[str, pd.DataFrame],
    ds: pd.DataFrame,
    preds: pd.DataFrame,
    thr: float,
    cost_bps: float,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> dict[str, BacktestResult]:
    """Run all strategies on one split; every signal date must lie in [start, end]."""
    syms = sorted(ds["symbol"].unique())
    p = preds.pivot(index="date", columns="symbol", values="prob")[syms].dropna()
    check_period(pd.DatetimeIndex(p.index), start, end, "predictions")
    reb = rebalance_dates(pd.DatetimeIndex(p.index), HORIZON)
    p = p.loc[reb]
    mom = ds.pivot(index="date", columns="symbol", values="ret_126")[syms].reindex(p.index)
    out = {
        "model": backtest(prices, _weights(p >= thr), cost_bps, end),
        "momentum_126d": backtest(prices, _weights(mom > 0), cost_bps, end),
        "equal_weight": backtest(prices, _weights(p.notna()), cost_bps, end),
    }
    spy_t = pd.DataFrame({BENCHMARK: 1.0}, index=p.index[:1])
    out["spy_buy_hold"] = backtest({BENCHMARK: prices[BENCHMARK]}, spy_t, cost_bps, end)
    return out


def run_backtests(
    data_root: Path, model_dir: Path, ds: pd.DataFrame, cost_bps: float = 10.0
) -> dict[str, Any]:
    meta = json.loads((model_dir / "metadata.json").read_text())
    preds = pd.read_csv(model_dir / "predictions.csv", parse_dates=["date"])
    symbols = meta["symbols"]
    dev_end, test_start = pd.Timestamp(meta["dev_end"]), pd.Timestamp(meta["test_start"])
    if dev_end >= test_start:
        raise ValueError("Metadata dev_end must precede test_start")
    bounds = {"dev_oof": (None, dev_end), "test": (test_start, None)}
    prices = {s: load_symbol(data_root, s) for s in symbols + [BENCHMARK]}
    result: dict[str, Any] = {
        "cost_bps_one_way": cost_bps,
        "rebalance_every_sessions": HORIZON,
        "threshold": meta["threshold"],
        "model_version": meta["version"],
        "assumptions": [
            "Signals use close(t) data; fills at next-session adjusted open.",
            "Adjusted prices: dividends reinvested; splits handled. Cash earns 0%.",
            "Costs charged on traded notional; 10 bps one-way is an assumption.",
            "Universe is current large caps: survivorship bias inflates all strategies.",
        ],
        "periods": {},
        "equity_curves": {},
    }
    for split in ("dev_oof", "test"):
        sp = preds[preds["split"] == split]
        lo, hi = bounds[split]
        res = run_split(prices, ds, sp, meta["threshold"], cost_bps, lo, hi)
        eq = res["model"].equity
        result["periods"][split] = {
            "start_bound": str(lo.date()) if lo is not None else None,
            "end_bound": str(hi.date()) if hi is not None else None,
            "first_signal": str(sp["date"].min().date()),
            "last_signal": str(sp["date"].max().date()),
            "equity_first": str(eq.index[0].date()),
            "equity_last": str(eq.index[-1].date()),
        }
        result[split] = {k: v.metrics for k, v in res.items()}
        result["equity_curves"][split] = {
            k: {str(pd.Timestamp(str(d)).date()): round(float(v), 6) for d, v in r.equity.items()}
            for k, r in res.items()
        }
    return result


if __name__ == "__main__":
    from marketops.training.cli import load_dataset

    root = Path("artifacts/models")
    mdir = root / (root / "LATEST").read_text().strip()
    meta = json.loads((mdir / "metadata.json").read_text())
    data = load_dataset(Path("data/raw"), meta["symbols"])
    res = run_backtests(Path("data/raw"), mdir, data)
    (mdir / "backtest.json").write_text(json.dumps(res))
    for split in ("dev_oof", "test"):
        print(split)
        for k, m in res[split].items():
            print(
                f"  {k:14s} ret={m['total_return']:+.1%} cagr={m['cagr']:+.1%} "
                f"vol={m['ann_vol']:.1%} sharpe={m['sharpe']:.2f} mdd={m['max_drawdown']:.1%} "
                f"turn={m['annual_turnover']:.1f}x cost={m['total_cost_frac']:.2%}"
            )

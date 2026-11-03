"""Leakage, purging and accounting tests on synthetic prices (no network, no real data)."""

import numpy as np
import pandas as pd
import pytest

from marketops.backtest.engine import backtest, performance
from marketops.features.build import FEATURES, HORIZON, build_dataset, symbol_features
from marketops.training import pipeline


def synth(n: int = 600, seed: int = 0, drift: float = 0.0005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, 0.01, n)
    close = 100 * np.exp(np.cumsum(r))
    opn = close * (1 + rng.normal(0, 0.002, n))
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2018-01-01", periods=n),
            "open": opn,
            "high": np.maximum(opn, close) * 1.002,
            "low": np.minimum(opn, close) * 0.998,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1_000, 5_000, n).astype(float),
            "dividend": 0.0,
            "split": 1.0,
        }
    )


def test_features_do_not_use_future_data() -> None:
    full, bench = synth(seed=1), synth(seed=2)
    cut = 450
    a = symbol_features(full, bench).set_index("date")
    b = symbol_features(full.iloc[:cut], bench.iloc[:cut]).set_index("date")
    cols = FEATURES
    common = b.index
    pd.testing.assert_frame_equal(a.loc[common, cols], b[cols], check_exact=False, rtol=1e-12)


def test_label_is_forward_and_label_end_is_h_sessions_ahead() -> None:
    f = synth(seed=3)
    out = symbol_features(f, synth(seed=4))
    i = 100
    expect = f["adj_close"].iloc[i + HORIZON] / f["adj_close"].iloc[i] - 1
    assert out["fwd_ret"].iloc[i] == pytest.approx(expect)
    assert out["label_end"].iloc[i] == f["date"].iloc[i + HORIZON]
    assert out["label"].tail(HORIZON).isna().all()


def test_purged_train_has_no_label_overlap_with_validation() -> None:
    ds = build_dataset({"A": synth(seed=5), "B": synth(seed=6), "SPY": synth(seed=7)})
    start = ds["date"].iloc[len(ds) // 2]
    tr = pipeline.purged_train(ds, start)
    assert (tr["date"] < start).all()
    assert (tr["label_end"] < start).all()
    assert len(tr) < len(ds[ds["date"] < start])  # some rows were purged


def test_scaler_is_fit_on_training_rows_only() -> None:
    ds = build_dataset({"A": synth(seed=5), "B": synth(seed=6), "SPY": synth(seed=7)})
    tr = ds.iloc[: len(ds) // 2].dropna(subset=["label"])
    m = pipeline.make_logreg()
    m.fit(tr[FEATURES], tr["label"])
    np.testing.assert_allclose(m.named_steps["scale"].mean_, tr[FEATURES].mean().to_numpy())


def test_metrics_confusion_and_constant_scores() -> None:
    y = np.array([1, 1, 0, 0, 1])
    m = pipeline.metrics(y, np.array([0.9, 0.2, 0.1, 0.8, 0.7]), 0.5)
    assert m["confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 2}
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert pipeline.metrics(y, np.full(5, 0.6), 0.5)["roc_auc"] is None


# ---------------- backtest accounting & timing


def flat_prices(opens: list[float], closes: list[float]) -> pd.DataFrame:
    n = len(opens)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=n),
            "open": opens,
            "close": closes,
            "adj_close": closes,
        }
    )


def test_signal_fills_at_next_open_not_signal_bar() -> None:
    px = flat_prices([10, 20, 20, 20], [10, 20, 22, 24])
    dates = px["date"]
    t = pd.DataFrame({"A": [1.0]}, index=[dates[0]])  # signal at close of day 0
    res = backtest({"A": px}, t, cost_bps=0)
    # Fills at day-1 open (20), not day-0 close (10): return = 24/20 - 1 = 20%
    assert res.equity.iloc[-1] == pytest.approx(1.2)


def test_costs_charged_on_traded_notional() -> None:
    px = flat_prices([10, 10, 10], [10, 10, 10])
    t = pd.DataFrame({"A": [1.0]}, index=[px["date"][0]])
    res = backtest({"A": px}, t, cost_bps=10)
    assert res.total_cost == pytest.approx(0.001)
    assert res.equity.iloc[-1] == pytest.approx(0.999)


def test_future_prices_do_not_change_earlier_equity() -> None:
    a = flat_prices([10, 11, 12, 13, 14], [10.5, 11.5, 12.5, 13.5, 14.5])
    b = a.copy()
    b.loc[4, ["open", "close", "adj_close"]] = [99, 99, 99]  # change only the last day
    t = pd.DataFrame({"A": [1.0]}, index=[a["date"][0]])
    ea, eb = backtest({"A": a}, t, 5).equity, backtest({"A": b}, t, 5).equity
    pd.testing.assert_series_equal(ea.iloc[:-1], eb.iloc[:-1])


def test_cash_and_weight_validation() -> None:
    px = flat_prices([10] * 4, [10] * 4)
    zero = pd.DataFrame({"A": [0.0]}, index=[px["date"][0]])
    assert backtest({"A": px}, zero, 10).equity.iloc[-1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        backtest({"A": px}, pd.DataFrame({"A": [1.5]}, index=[px["date"][0]]), 0)
    with pytest.raises(ValueError):
        backtest({"A": px}, pd.DataFrame({"A": [-0.1]}, index=[px["date"][0]]), 0)


def test_two_asset_rebalance_turnover() -> None:
    a = flat_prices([10] * 6, [10] * 6)
    b = flat_prices([10] * 6, [10] * 6)
    d = a["date"]
    t = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=[d[0], d[2]])
    res = backtest({"A": a, "B": b}, t, cost_bps=0)
    assert res.trades["delta_weight"].abs().sum() == pytest.approx(1.0 + 2.0)
    assert res.equity.iloc[-1] == pytest.approx(1.0)


def test_performance_metrics() -> None:
    eq = pd.Series([1.0, 1.1, 0.99, 1.2], index=pd.bdate_range("2024-01-01", periods=4))
    m = performance(eq)
    assert m["total_return"] == pytest.approx(0.2)
    assert m["max_drawdown"] == pytest.approx(0.99 / 1.1 - 1)

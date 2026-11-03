"""Period separation: development results must never be marked on held-out prices."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marketops.backtest.engine import backtest
from marketops.backtest.run import check_period, run_backtests, run_split
from marketops.data.store import BENCHMARK
from marketops.features.build import FEATURES

DEV_END = pd.Timestamp("2021-12-31")
TEST_START = pd.Timestamp("2022-01-01")


def prices(seed: int = 0, jump_after: pd.Timestamp | None = None) -> pd.DataFrame:
    dates = pd.bdate_range("2021-10-01", "2022-06-30")
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(dates))))
    if jump_after is not None:  # a huge move that only exists in the held-out period
        close = np.where(dates > jump_after, close * 3, close)
    return pd.DataFrame({"date": dates, "open": close, "close": close, "adj_close": close})


def test_end_cuts_equity_curve_and_ignores_later_prices() -> None:
    a, b = prices(), prices(jump_after=DEV_END)
    t = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-12-23")])
    ra, rb = backtest({"A": a}, t, 0, DEV_END), backtest({"A": b}, t, 0, DEV_END)
    assert ra.equity.index[-1] <= DEV_END
    pd.testing.assert_series_equal(ra.equity, rb.equity)  # test-period prices have no effect
    # Without the boundary the curve leaks into the test period (the original defect).
    assert backtest({"A": a}, t, 0).equity.index[-1] > DEV_END


def test_signal_executing_after_end_is_dropped_not_shifted() -> None:
    px = prices()
    t = pd.DataFrame({"A": [1.0, 1.0]}, index=[pd.Timestamp("2021-12-23"), DEV_END])
    res = backtest({"A": px}, t, 0, DEV_END)
    assert res.trades["date"].max() <= DEV_END  # 12-31 signal would fill on 2022-01-03
    only_late = t.iloc[1:]
    with pytest.raises(ValueError, match="No executable"):
        backtest({"A": px}, only_late, 0, DEV_END)


def test_check_period_rejects_overlapping_dates() -> None:
    dev = pd.DatetimeIndex(["2021-12-23", "2022-01-03"])
    with pytest.raises(ValueError, match="after period end"):
        check_period(dev, None, DEV_END, "predictions")
    test = pd.DatetimeIndex(["2021-12-30", "2022-01-03"])
    with pytest.raises(ValueError, match="before period start"):
        check_period(test, TEST_START, None, "predictions")
    check_period(pd.DatetimeIndex(["2021-12-23"]), None, DEV_END, "predictions")


def _fixture(tmp_path: Path, dev_end: str, test_start: str) -> tuple[Path, Path, pd.DataFrame]:
    data, model = tmp_path / "data", tmp_path / "model"
    data.mkdir()
    model.mkdir()
    for i, s in enumerate(["AAA", BENCHMARK]):
        prices(i).assign(
            volume=1,
            high=lambda d: d["close"],
            low=lambda d: d["close"],
            dividends=0.0,
            splits=1.0,
        ).to_csv(data / f"{s}.csv", index=False)
    dates = pd.bdate_range("2021-10-01", "2022-06-30")
    rows = [
        {"date": d, "symbol": "AAA", "label": 1, "fwd_ret": 0.0, "prob": 0.6,
         "split": "dev_oof" if d <= DEV_END else "test"}
        for d in dates
    ]  # fmt: skip
    pd.DataFrame(rows).to_csv(model / "predictions.csv", index=False)
    meta = {
        "symbols": ["AAA"], "threshold": 0.5, "version": "v",
        "dev_end": dev_end, "test_start": test_start,
    }  # fmt: skip
    (model / "metadata.json").write_text(json.dumps(meta))
    ds = pd.DataFrame({"date": dates, "symbol": "AAA", "ret_126": 0.1})
    for f in FEATURES:
        ds[f] = 0.0
    return data, model, ds


def test_run_backtests_dev_curve_ends_at_dev_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, model, ds = _fixture(tmp_path, "2021-12-31", "2022-01-01")
    monkeypatch.setattr(
        "marketops.backtest.run.load_symbol",
        lambda root, s: pd.read_csv(root / f"{s}.csv", parse_dates=["date"]),
    )
    res = run_backtests(data, model, ds)
    for curve in res["equity_curves"]["dev_oof"].values():
        assert max(curve) <= "2021-12-31"
    per = res["periods"]
    assert per["dev_oof"]["equity_last"] <= "2021-12-31" and per["dev_oof"]["last_signal"] < "2022"
    assert per["test"]["first_signal"] >= "2022-01-01"
    again = run_backtests(data, model, ds)  # reproducible
    assert again["equity_curves"] == res["equity_curves"]


def test_mismatched_metadata_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "marketops.backtest.run.load_symbol",
        lambda root, s: pd.read_csv(root / f"{s}.csv", parse_dates=["date"]),
    )
    # dev_end earlier than the saved dev predictions -> overlap must raise, not be evaluated
    data, model, ds = _fixture(tmp_path, "2021-11-30", "2022-01-01")
    with pytest.raises(ValueError, match="after period end"):
        run_backtests(data, model, ds)
    # dev_end not before test_start
    (model / "metadata.json").write_text(
        json.dumps(
            {
                "symbols": ["AAA"],
                "threshold": 0.5,
                "version": "v",
                "dev_end": "2022-01-01",
                "test_start": "2022-01-01",
            }
        )  # fmt: skip
    )
    with pytest.raises(ValueError, match="must precede"):
        run_backtests(data, model, ds)


def test_run_split_rejects_test_dates_in_dev(tmp_path: Path) -> None:
    px = {"AAA": prices(), BENCHMARK: prices(1)}
    ds = pd.DataFrame({"symbol": ["AAA"], "ret_126": [0.1]})
    d = pd.bdate_range("2021-12-20", "2022-01-10")
    preds = pd.DataFrame({"date": d, "symbol": "AAA", "prob": 0.6})
    with pytest.raises(ValueError, match="after period end"):
        run_split(px, ds.assign(date=d[0]), preds, 0.5, 0.0, None, DEV_END)

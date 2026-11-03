"""Regression tests for defects found in the technical audit (see docs/technical-audit.md)."""

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from marketops.backtest.engine import backtest, performance
from marketops.config import get_settings
from marketops.data.store import cache_path, fingerprint, ingest_symbol, write_manifest
from marketops.data.validation import DataQualityError
from marketops.features.build import FEATURES, build_dataset
from marketops.inference import service
from marketops.training import pipeline
from tests.test_ml_backtest import flat_prices, synth

SUMMARY_METRICS = ("roc_auc", "pr_auc", "brier", "log_loss", "precision", "recall", "f1")


def long_synth(seed: int) -> pd.DataFrame:
    f = synth(n=2800, seed=seed)
    f["date"] = pd.bdate_range("2012-01-02", periods=len(f))
    return f


@pytest.fixture
def trained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Genuine pipeline.run() output (logreg only, small) on synthetic data spanning DEV_END."""
    data, models = tmp_path / "raw", tmp_path / "models"
    data.mkdir()
    frames = {"AAA": long_synth(1), "BBB": long_synth(3), "SPY": long_synth(2)}
    for s, f in frames.items():
        f.to_csv(cache_path(data, s), index=False, date_format="%Y-%m-%d")
    monkeypatch.setattr(pipeline, "CANDIDATES", {"logreg": lambda: pipeline.make_logreg(0.01)})
    monkeypatch.setattr(pipeline, "FIRST_FOLD_YEAR", 2016)
    res = pipeline.run(
        build_dataset(frames), models, {"fingerprint": fingerprint(data, list(frames))}
    )
    assert res.model_dir.name == (models / "LATEST").read_text()
    return data, models


def point_api_at(monkeypatch: pytest.MonkeyPatch, data: Path, models: Path) -> None:
    monkeypatch.setenv("MARKETOPS_DATA_DIR", str(data))
    monkeypatch.setenv("MARKETOPS_MODEL_DIR", str(models))
    get_settings.cache_clear()


# ---------------- ML validity


def test_dev_labels_never_reach_into_test_period() -> None:
    ds = build_dataset({"A": long_synth(1), "SPY": long_synth(2)})
    raw = pipeline.labelled(ds[ds["date"] <= pipeline.DEV_END])
    assert (raw["label_end"] > pipeline.DEV_END).any()  # the old behaviour kept these rows
    dev = pipeline.dev_labelled(ds)
    assert (dev["label_end"] <= pipeline.DEV_END).all() and len(dev) < len(raw)


def test_training_is_deterministic_and_metadata_matches_artifact(
    trained: tuple[Path, Path], tmp_path: Path
) -> None:
    data, models = trained
    d = models / (models / "LATEST").read_text()
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["model_sha256"] == hashlib.sha256((d / "model.joblib").read_bytes()).hexdigest()
    assert meta["features"] == FEATURES and meta["version"] == d.name
    assert meta["data"]["fingerprint"]["AAA"]["rows"] == 2800
    assert max(meta["config"]["candidates"]) == "logreg" and meta["config"]["seed"] == 42
    frames = {
        s: pd.read_csv(cache_path(data, s), parse_dates=["date"]) for s in ("AAA", "BBB", "SPY")
    }
    again = pipeline.run(build_dataset(frames), tmp_path / "again")
    pd.testing.assert_frame_equal(
        pd.read_csv(d / "predictions.csv"), pd.read_csv(again.model_dir / "predictions.csv")
    )


# ---------------- inference / API with genuine artefacts


def test_api_serves_genuine_saved_artifacts(
    trained: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data, models = trained
    point_api_at(monkeypatch, data, models)
    lm = service.load_latest(models)
    with TestClient(app) as c:
        body = c.get("/analysis/AAA").json()
        direct = float(lm.model.predict_proba(pd.DataFrame([body["features"]])[FEATURES])[0, 1])
        assert body["probability_up"] == pytest.approx(direct)
        assert body["model_version"] == lm.version
        assert c.get("/model/metadata").json()["version"] == lm.version
        assert c.get("/predictions/AAA?split=dev_oof").status_code == 200
        assert c.get("/predictions/AAA?split=bogus").status_code == 422
    get_settings.cache_clear()


def _tamper(models: Path, **meta_changes: Any) -> None:
    d = models / (models / "LATEST").read_text()
    meta = json.loads((d / "metadata.json").read_text())
    meta.update(meta_changes)
    (d / "metadata.json").write_text(json.dumps(meta))


@pytest.mark.parametrize(
    "change",
    [
        {"model_sha256": "0" * 64},
        {"model_sha256": None},
        {"features": FEATURES[:-1]},
        {"version": "someone_else"},
        {"env": {"sklearn": "0.0.1"}},
    ],
)
def test_inference_rejects_mismatched_artifacts(
    trained: tuple[Path, Path], change: dict[str, Any]
) -> None:
    _, models = trained
    _tamper(models, **change)
    with pytest.raises(service.ModelNotAvailable):
        service.load_latest(models)


def test_inference_rejects_bad_latest_pointer(trained: tuple[Path, Path]) -> None:
    _, models = trained
    (models / "LATEST").write_text("../escape")
    with pytest.raises(service.ModelNotAvailable):
        service.load_latest(models)


def test_api_failure_states_are_503_not_500(
    trained: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data, models = trained
    (models / (models / "LATEST").read_text() / "predictions.csv").unlink()
    cache_path(data, "BBB").unlink()
    point_api_at(monkeypatch, data, models)
    with TestClient(app, raise_server_exceptions=False) as c:
        assert c.get("/predictions/AAA").status_code == 503
        assert c.get("/monitoring").status_code == 503
        assert c.get("/analysis/BBB").status_code == 503
        assert c.get("/analysis/NOPE").status_code == 404
    get_settings.cache_clear()


# ---------------- data provenance


def _frame(adj_scale: float = 1.0, start: str = "2024-01-01", n: int = 10) -> pd.DataFrame:
    f = synth(n=n, seed=0)
    f["date"] = pd.bdate_range(start, periods=n)
    f["adj_close"] = f["close"] * adj_scale
    return f


class _Fixed:
    name = "fixed"

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        return self.df.copy()


def test_ingest_refuses_splice_across_changed_adjustment(tmp_path: Path) -> None:
    window = (date(2024, 1, 1), date(2025, 1, 1))
    ingest_symbol(_Fixed(_frame()), tmp_path, "A", *window)
    # Later partial window after the provider rescaled adj_close (a dividend was paid).
    later = _Fixed(_frame(adj_scale=0.98, start="2024-01-08"))
    with pytest.raises(DataQualityError, match="re-ingest the full history"):
        ingest_symbol(later, tmp_path, "A", *window)
    # A full-history refresh (fresh covers every cached date) is allowed.
    full = _Fixed(_frame(adj_scale=0.98, n=12))
    frame, _ = ingest_symbol(full, tmp_path, "A", *window)
    assert len(frame) == 12


def test_manifest_partial_reingest_keeps_other_symbols(tmp_path: Path) -> None:
    write_manifest(tmp_path, "p", {"A": _frame(), "B": _frame()})
    write_manifest(tmp_path, "p", {"A": _frame(n=12)})
    syms = json.loads((tmp_path / "manifest.json").read_text())["symbols"]
    assert set(syms) == {"A", "B"} and syms["A"]["rows"] == 12


# ---------------- backtest accounting


def test_dividend_drop_in_close_is_not_double_counted() -> None:
    # Raw close falls 5 at the ex-date but adj_close (total return) is continuous: the equity
    # curve must follow adj_close, neither losing the gap nor adding the dividend again.
    adj = np.array([100.0, 101, 102, 103, 104, 105])
    close = adj.copy()
    close[3:] -= 5
    px = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=6),
            "open": close,
            "close": close,
            "adj_close": adj,
        }
    )
    t = pd.DataFrame({"A": [1.0]}, index=[px["date"][0]])
    res = backtest({"A": px}, t, cost_bps=0)
    assert res.equity.iloc[-1] == pytest.approx(105 / 101)  # filled at day-1 adjusted open


def test_signal_on_non_trading_day_executes_next_session() -> None:
    px = flat_prices([10] * 10, [10] * 10)  # Mon 2024-01-01 .. Fri 2024-01-12
    saturday = pd.Timestamp("2024-01-06")
    res = backtest({"A": px}, pd.DataFrame({"A": [1.0]}, index=[saturday]), 0)
    assert res.trades["date"].iloc[0] == pd.Timestamp("2024-01-08")


def test_missing_price_or_weight_is_rejected() -> None:
    px = flat_prices([10] * 4, [10] * 4)
    bad = px.copy()
    bad.loc[2, "open"] = np.nan
    t = pd.DataFrame({"A": [1.0]}, index=[px["date"][0]])
    with pytest.raises(ValueError, match="Missing or non-positive"):
        backtest({"A": bad}, t, 0)
    with pytest.raises(ValueError, match="missing weights"):
        backtest({"A": px}, pd.DataFrame({"A": [np.nan]}, index=[px["date"][0]]), 0)


def test_multi_asset_common_calendar_and_cost_scales_with_turnover() -> None:
    a = flat_prices([10] * 8, [10] * 8)
    b = flat_prices([10] * 8, [10] * 8).drop(index=3)  # B has no bar on a[3]
    d = a["date"]
    t = pd.DataFrame({"A": [0.5, 0.0], "B": [0.5, 0.5]}, index=[d[0], d[4]])
    res = backtest({"A": a, "B": b}, t, cost_bps=10)
    assert d[3] not in res.equity.index  # a day missing for one asset is dropped for all
    # build trades 1.0 of NAV, rebalance trades 0.5: 1.5 x 10 bps
    assert res.total_cost == pytest.approx(0.0015, rel=0.01)


def test_cagr_sharpe_and_drawdown_known_values() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="365D")
    m = performance(pd.Series([1.0, 2.0, 1.0], index=idx))
    assert m["cagr"] == pytest.approx(0.0, abs=1e-9)
    assert m["max_drawdown"] == pytest.approx(-0.5)
    r = np.array([0.01, -0.01, 0.01, -0.01])
    eq = pd.Series(
        np.concatenate([[1.0], np.cumprod(1 + r)]), index=pd.bdate_range("2024-01-01", periods=5)
    )
    # Sharpe = mean/std(ddof=1) of daily returns x sqrt(252), risk-free rate 0.
    assert performance(eq)["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))


# ---------------- MLflow hygiene


def test_mlflow_artifacts_do_not_pollute_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mlflow")
    from marketops.training.tracking import log_run

    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    d = tmp_path / "m"
    d.mkdir()
    (d / "metadata.json").write_text("{}")
    (d / "model.joblib").write_bytes(b"x")
    summary: dict[str, Any] = {
        "version": "t",
        "selected_model": "m",
        "threshold": 0.5,
        "dev_end": "x",
        "test_start": "y",
        "symbols": ["A"],
        "train_rows_final": 1,
        "env": {},
        "walk_forward_cv": {},
        "test": {"model": dict.fromkeys(SUMMARY_METRICS, 0.5)},
        "latency": {},
    }
    log_run(summary, d, tmp_path / "db" / "mlflow.db")
    assert not (work / "mlruns").exists()
    assert (tmp_path / "db" / "mlartifacts").exists()


# ---------------- PostgreSQL (opt-in, real database)


@pytest.mark.skipif(
    os.getenv("MARKETOPS_RUN_INTEGRATION") != "1",
    reason="Set MARKETOPS_RUN_INTEGRATION=1 with MARKETOPS_DATABASE_URL for real DB checks",
)
def test_postgres_prediction_and_registry_round_trip(trained: tuple[Path, Path]) -> None:
    from marketops.persistence import database, models

    engine = database.get_engine()
    lm = service.load_latest(trained[1])
    models.register_model(engine, lm.meta)
    models.register_model(engine, lm.meta)  # idempotent
    row = {
        "model_version": lm.version,
        "symbol": "ZZTEST",
        "as_of_date": "2024-01-02",
        "probability_up": 0.51,
        "threshold": 0.5,
        "signal": "above_threshold",
        "features": {"ret_1": 0.1},
    }
    assert models.log_prediction(engine, row) is True
    got = [r for r in models.recent_predictions(engine, 20) if r["symbol"] == "ZZTEST"]
    assert got and got[0]["probability_up"] == pytest.approx(0.51)
    assert got[0]["features"] == {"ret_1": 0.1}

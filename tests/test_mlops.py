"""Monitoring, SHAP and MLflow tests (optional deps are skipped when absent)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marketops.features.build import FEATURES, build_dataset
from marketops.monitoring.checks import data_quality, drift_report, psi
from marketops.training import pipeline
from tests.test_ml_backtest import synth


def test_psi_zero_for_same_and_large_for_shift() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi(a, rng.normal(2, 1, 5000)) > 0.25


def test_data_quality_flags_stale_and_jumps() -> None:
    f = synth(n=50)
    f.loc[30, "adj_close"] *= 1.5
    out = data_quality({"A": f}, today=f["date"].max().date() + pd.Timedelta(days=10))
    assert out["A"]["stale"] is True and out["A"]["moves_over_25pct"] >= 1


def test_drift_report_levels() -> None:
    ds = build_dataset({"A": synth(seed=1), "SPY": synth(seed=2)})
    ref, rec = ds.iloc[:300], ds.iloc[-60:].assign(vol_21=lambda d: d["vol_21"] * 5)
    rep = drift_report(ref, rec, np.full(300, 0.5) + np.arange(300) * 1e-4, np.full(60, 0.55))
    assert rep["feature_psi"]["vol_21"]["level"] == "alert"
    assert set(rep["feature_psi"]) == set(FEATURES)


def test_shap_matches_logit_for_linear_model() -> None:
    pytest.importorskip("shap")
    from marketops.explainability.shap_explain import explain

    ds = build_dataset({"A": synth(seed=1), "B": synth(seed=3), "SPY": synth(seed=2)})
    ds = ds.dropna(subset=["label"])
    m = pipeline.make_logreg()
    m.fit(ds[FEATURES], ds["label"])
    rows, bg_rows = ds.iloc[:20], ds.iloc[:100]  # shap caps linear backgrounds at 100 rows
    sv = explain(m, bg_rows, rows)
    clf = m.named_steps["clf"]
    bg = m.named_steps["scale"].transform(bg_rows[FEATURES])
    logit = m.named_steps["scale"].transform(rows[FEATURES]) @ clf.coef_[0] + clf.intercept_[0]
    base = (bg @ clf.coef_[0]).mean() + clf.intercept_[0]
    # additivity: baseline + sum of SHAP contributions == model log-odds
    np.testing.assert_allclose(sv.sum(axis=1) + base, logit, atol=1e-6)


def test_mlflow_logs_run(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    from marketops.training.tracking import log_run

    summary = {
        "version": "t1",
        "selected_model": "logreg",
        "threshold": 0.5,
        "dev_end": "x",
        "test_start": "y",
        "symbols": ["A"],
        "train_rows_final": 10,
        "env": {"python": "3"},
        "walk_forward_cv": {"logreg": {"oof_log_loss": 0.69}},
        "test": {
            "model": {
                k: 0.5
                for k in ("roc_auc", "pr_auc", "brier", "log_loss", "precision", "recall", "f1")
            }
        },
        "latency": {"single_row_ms_p50": 1.0},
    }
    d = tmp_path / "m"
    d.mkdir()
    (d / "metadata.json").write_text("{}")
    (d / "model.joblib").write_bytes(b"x")
    run_id = log_run(summary, d, tmp_path / "mlflow.db")
    run = mlflow.get_run(run_id)
    assert run.data.metrics["test.roc_auc"] == 0.5 and run.data.params["selected_model"] == "logreg"

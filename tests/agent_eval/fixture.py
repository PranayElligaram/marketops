"""Deterministic model/data artefacts shared by agent tests and the evaluation harness.

Metric values mirror the audited layout of a real model directory but are fixed constants, so
expected comparison outcomes in the evaluation set never depend on a retrained model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib

from marketops.config import Settings
from marketops.data.store import cache_path
from marketops.features.build import FEATURES, build_dataset
from marketops.training import pipeline
from tests.test_ml_backtest import synth

METRICS = {"n": 100, "roc_auc": 0.4961, "pr_auc": 0.5571, "brier": 0.2483, "log_loss": 0.6898}
BASE = {"n": 100, "roc_auc": 0.5, "pr_auc": 0.55, "brier": 0.2477, "log_loss": 0.6885}


def _bt(cagr: float, sharpe: float, ret: float, vol: float, dd: float, turn: float) -> dict:
    return {
        "total_return": ret,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "years": 4.8,
        "annual_turnover": turn,
        "total_cost_frac": 0.01,
    }


BACKTEST = {
    "cost_bps_one_way": 10,
    "rebalance_every_sessions": 5,
    "assumptions": ["Costs are an assumption.", "Universe has survivorship bias."],
    "equity_curves": {
        "test": {"model": {"2022-01-03": 1.0, "2026-10-26": 1.9}},
        "dev_oof": {"model": {"2010-01-04": 1.0, "2021-12-31": 2.5}},
    },
    "test": {
        "model": _bt(0.1439, 0.9243, 0.9108, 0.1599, -0.1603, 34.59),
        "momentum_126d": _bt(0.1610, 1.0932, 1.0525, 0.1470, -0.2419, 11.34),
        "equal_weight": _bt(0.1729, 1.1682, 1.1557, 0.1462, -0.1739, 1.28),
        "spy_buy_hold": _bt(0.1186, 0.7385, 0.7153, 0.1724, -0.2481, 0.18),
    },
    "dev_oof": {
        "model": _bt(0.05, 0.4, 0.3, 0.2, -0.3, 30.0),
        "equal_weight": _bt(0.12, 0.8, 1.2, 0.18, -0.35, 1.0),
    },
}


def build_settings(tmp_path: Path) -> Settings:
    data, models_dir = tmp_path / "raw", tmp_path / "models"
    data.mkdir()
    frames = {"AAA": synth(seed=1), "BBB": synth(seed=3, drift=-0.0002), "SPY": synth(seed=2)}
    for s, f in frames.items():
        f.to_csv(cache_path(data, s), index=False, date_format="%Y-%m-%d")
    ds = build_dataset(frames).dropna(subset=["label"])
    model = pipeline.make_logreg()
    model.fit(ds[FEATURES], ds["label"])
    v = models_dir / "v1"
    v.mkdir(parents=True)
    joblib.dump(model, v / "model.joblib")
    meta = {
        "version": "v1",
        "selected_model": "logreg",
        "threshold": 0.5,
        "dev_end": "2021-12-31",
        "test_start": "2022-01-01",
        "model_sha256": hashlib.sha256((v / "model.joblib").read_bytes()).hexdigest(),
        "symbols": ["AAA", "BBB"],
        "features": FEATURES,
        "walk_forward_cv": {
            "logreg_C0.01": {"oof_roc_auc": 0.5139, "fold_auc_mean": 0.5166, "fold_auc_min": 0.4931}
        },
        "dev_baselines": {},
        "test": {"model": {**METRICS, "calibration": []}, "baseline_always_up": BASE},
        "caveats": ["Universe is tiny: survivorship bias."],
    }
    (v / "metadata.json").write_text(json.dumps(meta))
    (v / "backtest.json").write_text(json.dumps(BACKTEST))
    (v / "shap.json").write_text(
        json.dumps({"mean_abs_shap_logodds": {"ret_5": 0.043, "vol_21": 0.04}})
    )
    (models_dir / "LATEST").write_text("v1")
    return Settings(_env_file=None, data_dir=str(data), model_dir=str(models_dir))

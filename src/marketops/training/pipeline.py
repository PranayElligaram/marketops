"""Supervised training/evaluation: P(5-session forward return > 0).

Protocol
- Development period: dates <= DEV_END. Held-out test: dates >= TEST_START. Test is touched
  exactly once, after model choice and threshold are frozen.
- Model selection: expanding-window walk-forward over development years. For each fold the
  training rows are purged: any row whose label_end >= fold start is dropped (label overlap),
  so no training label depends on data inside the validation window.
- Preprocessing (scaling) lives inside the sklearn Pipeline and is fit on training rows only.
- Threshold: chosen on out-of-fold development predictions (balanced accuracy).
- Final model: refit on all purged development rows, then evaluated once on the test period.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from marketops.features.build import FEATURES, HORIZON

DEV_END = pd.Timestamp("2021-12-31")
TEST_START = pd.Timestamp("2022-01-01")
FIRST_FOLD_YEAR = 2010  # first validation year; needs >= ~5y of earlier history
SEED = 42


def make_logreg(c: float = 1.0) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=c, max_iter=1000))])


def make_xgb(depth: int, n: int) -> Pipeline:
    clf = XGBClassifier(
        n_estimators=n,
        max_depth=depth,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=200,
        reg_lambda=10.0,
        random_state=SEED,
        n_jobs=4,
        eval_metric="logloss",
    )
    return Pipeline([("clf", clf)])


CANDIDATES: dict[str, Any] = {
    "logreg_C0.01": lambda: make_logreg(0.01),
    "logreg_C1": lambda: make_logreg(1.0),
    "xgb_d2_n200": lambda: make_xgb(2, 200),
    "xgb_d3_n300": lambda: make_xgb(3, 300),
}


def labelled(ds: pd.DataFrame) -> pd.DataFrame:
    return ds.dropna(subset=["label", "label_end"])


def dev_labelled(ds: pd.DataFrame) -> pd.DataFrame:
    """Development rows whose label is fully realised by DEV_END.

    Rows dated near DEV_END whose 5-session label reaches into the test period are excluded, so
    no development label (hence no CV score, model choice or threshold) depends on test prices.
    """
    dev = labelled(ds[ds["date"] <= DEV_END])
    return dev[dev["label_end"] <= DEV_END]


def purged_train(ds: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """Rows usable for training before `start`: date < start and label fully known before it."""
    lab = labelled(ds)
    return lab[(lab["date"] < start) & (lab["label_end"] < start)]


def walk_forward(ds: pd.DataFrame, factory: Any) -> pd.DataFrame:
    """Out-of-fold predictions for each dev year (expanding window, purged)."""
    dev = dev_labelled(ds)
    preds = []
    for year in range(FIRST_FOLD_YEAR, DEV_END.year + 1):
        lo, hi = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
        tr = purged_train(dev, lo)
        va = dev[(dev["date"] >= lo) & (dev["date"] <= hi)]
        m: BaseEstimator = factory()
        m.fit(tr[FEATURES], tr["label"])
        p = va[["date", "symbol", "label", "fwd_ret"]].copy()
        p["prob"] = m.predict_proba(va[FEATURES])[:, 1]
        p["fold"] = year
        preds.append(p)
    return pd.concat(preds).reset_index(drop=True)


def metrics(y: np.ndarray, prob: np.ndarray, thr: float) -> dict[str, Any]:
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prob_c = np.clip(prob, 1e-6, 1 - 1e-6)
    out: dict[str, Any] = {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "threshold": float(thr),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))),
        "brier": float(brier_score_loss(y, prob_c)),
        "log_loss": float(log_loss(y, prob_c, labels=[0, 1])),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if len(np.unique(prob)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, prob))
        out["pr_auc"] = float(average_precision_score(y, prob))
    else:
        out["roc_auc"] = out["pr_auc"] = None
    return out


def calibration_table(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    edges = np.quantile(prob, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-12
    idx = np.clip(np.searchsorted(edges, prob, side="right") - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append(
                {
                    "bin": b,
                    "n": int(m.sum()),
                    "mean_pred": float(prob[m].mean()),
                    "observed": float(y[m].mean()),
                }
            )
    return rows


def best_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    grid = np.unique(np.quantile(prob, np.linspace(0.05, 0.95, 91)))
    scores = [metrics(y, prob, t)["balanced_accuracy"] for t in grid]
    return float(grid[int(np.argmax(scores))])


def subsample_nonoverlap(df: pd.DataFrame) -> pd.DataFrame:
    """Every HORIZON-th trading date: removes label overlap within a symbol."""
    dates = np.sort(df["date"].unique())
    keep = set(dates[::HORIZON])
    return df[df["date"].isin(keep)]


def momentum_prob(df: pd.DataFrame) -> np.ndarray:
    """Naive momentum 'probability': logistic of scaled 21d return (score only, uncalibrated)."""
    return np.asarray(1 / (1 + np.exp(-df["ret_21"].to_numpy() * 10)))


@dataclass
class TrainResult:
    model_dir: Path
    summary: dict[str, Any]


def run(
    ds: pd.DataFrame, out_root: Path, data_manifest: dict[str, Any] | None = None
) -> TrainResult:
    t0 = time.perf_counter()
    dev = dev_labelled(ds)
    base_rate_train = float(dev["label"].mean())

    # ---- walk-forward model selection (development data only)
    cv: dict[str, Any] = {}
    oof_by_model: dict[str, pd.DataFrame] = {}
    for name, factory in CANDIDATES.items():
        oof = walk_forward(ds, factory)
        oof_by_model[name] = oof
        y, p = oof["label"].to_numpy(), oof["prob"].to_numpy()
        per_fold = pd.Series(
            {str(k): roc_auc_score(g["label"], g["prob"]) for k, g in oof.groupby("fold")}
        )
        cv[name] = {
            "oof_log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
            "oof_brier": float(brier_score_loss(y, p)),
            "oof_roc_auc": float(roc_auc_score(y, p)),
            "fold_auc_mean": float(per_fold.mean()),
            "fold_auc_min": float(per_fold.min()),
            "folds_auc_above_0.5": int((per_fold.gt(0.5)).sum()),
            "n_folds": int(len(per_fold)),
        }
    # Dev baselines on the same OOF rows
    any_oof = next(iter(oof_by_model.values()))
    dev_rows = dev.merge(any_oof[["date", "symbol"]], on=["date", "symbol"])
    y_dev = dev_rows["label"].to_numpy()
    baselines_cv = {
        "always_up_train_rate": {
            "oof_log_loss": float(
                log_loss(y_dev, np.full(len(y_dev), base_rate_train), labels=[0, 1])
            ),
            "oof_brier": float(brier_score_loss(y_dev, np.full(len(y_dev), base_rate_train))),
        },
        "momentum_21d": {
            "oof_roc_auc": float(roc_auc_score(y_dev, dev_rows["ret_21"])),
        },
    }
    # Selection criterion: lowest OOF log-loss (proper scoring rule; threshold independent).
    best = min(cv, key=lambda k: cv[k]["oof_log_loss"])
    oof_best = oof_by_model[best]
    thr = best_threshold(oof_best["label"].to_numpy(), oof_best["prob"].to_numpy())

    # ---- final refit on purged development rows, single test evaluation
    final_train = purged_train(ds, TEST_START)
    model = CANDIDATES[best]()
    model.fit(final_train[FEATURES], final_train["label"])
    test = labelled(ds[ds["date"] >= TEST_START])
    prob = model.predict_proba(test[FEATURES])[:, 1]
    y = test["label"].to_numpy()

    # Baselines fitted on train only
    train_rate = float(final_train["label"].mean())
    mom_thr = 0.5
    test_report = {
        "model": {**metrics(y, prob, thr), "calibration": calibration_table(y, prob)},
        "baseline_always_up": metrics(y, np.full(len(y), train_rate), 0.0),
        "baseline_momentum_21d": metrics(y, momentum_prob(test), mom_thr),
    }
    # Always-up predicts positive for all rows: threshold 0 forces that; probs are constant.
    nono = subsample_nonoverlap(test.assign(prob=prob))
    test_report["model_nonoverlap_subsample"] = metrics(
        nono["label"].to_numpy(), nono["prob"].to_numpy(), thr
    )
    # Alternative candidate on the held-out set is reported for transparency, not selection.
    others: dict[str, Any] = {}
    for name, factory in CANDIDATES.items():
        if name == best:
            continue
        m = factory()
        m.fit(final_train[FEATURES], final_train["label"])
        pr = m.predict_proba(test[FEATURES])[:, 1]
        others[name] = {
            "roc_auc": float(roc_auc_score(y, pr)),
            "brier": float(brier_score_loss(y, pr)),
        }

    # Inference latency (single-row, warm)
    row = test[FEATURES].iloc[[0]]
    model.predict_proba(row)
    lat = []
    for _ in range(200):
        s = time.perf_counter()
        model.predict_proba(row)
        lat.append((time.perf_counter() - s) * 1000)
    latency = {
        "single_row_ms_p50": float(np.percentile(lat, 50)),
        "single_row_ms_p95": float(np.percentile(lat, 95)),
    }
    s = time.perf_counter()
    model.predict_proba(test[FEATURES])
    latency["batch_rows"] = int(len(test))
    latency["batch_ms"] = float((time.perf_counter() - s) * 1000)

    # ---- persist
    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    mdir = out_root / version
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, mdir / "model.joblib")
    model_sha256 = hashlib.sha256((mdir / "model.joblib").read_bytes()).hexdigest()
    # Predictions for backtesting: dev = out-of-fold, test = final model.
    oof_best.assign(split="dev_oof").drop(columns=["fold"]).to_csv(
        mdir / "predictions.csv", index=False
    )
    test_pred = test[["date", "symbol", "label", "fwd_ret"]].assign(prob=prob, split="test")
    test_pred.to_csv(mdir / "predictions.csv", mode="a", header=False, index=False)
    summary = {
        "version": version,
        "target": f"P(adjusted-close return over next {HORIZON} sessions > 0)",
        "selected_model": best,
        "features": FEATURES,
        "model_sha256": model_sha256,
        "config": {
            "horizon": HORIZON,
            "seed": SEED,
            "first_fold_year": FIRST_FOLD_YEAR,
            "candidates": list(CANDIDATES),
            "selection_criterion": "walk-forward OOF log-loss",
            "threshold_criterion": "balanced accuracy on development OOF",
        },
        "threshold": thr,
        "dev_end": str(DEV_END.date()),
        "test_start": str(TEST_START.date()),
        "train_rows_final": int(len(final_train)),
        "test_rows": int(len(test)),
        "symbols": sorted(ds["symbol"].unique().tolist()),
        "walk_forward_cv": cv,
        "dev_baselines": baselines_cv,
        "test": test_report,
        "test_other_candidates_not_for_selection": others,
        "latency": latency,
        "data": data_manifest,
        "env": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "train_seconds": round(time.perf_counter() - t0, 1),
        "caveats": [
            "Universe is 10 currently-large US stocks: survivorship bias.",
            "Overlapping 5-day labels reduce effective sample size; see nonoverlap subsample.",
            "No transaction costs in classification metrics.",
        ],
    }
    (mdir / "metadata.json").write_text(json.dumps(summary, indent=2))
    (out_root / "LATEST").write_text(version)
    return TrainResult(mdir, summary)

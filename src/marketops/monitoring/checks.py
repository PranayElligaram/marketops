"""Data-quality and drift monitoring against the reference (training) distribution."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from marketops.features.build import FEATURES

PSI_WARN, PSI_ALERT = 0.1, 0.25


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index using reference-quantile bins."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / max(len(current), 1)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _level(v: float) -> str:
    return "alert" if v >= PSI_ALERT else "warn" if v >= PSI_WARN else "ok"


def data_quality(frames: dict[str, pd.DataFrame], today: date | None = None) -> dict[str, Any]:
    today = today or date.today()
    out: dict[str, Any] = {}
    for sym, f in frames.items():
        last = f["date"].max().date()
        r = f["adj_close"].pct_change().abs()
        out[sym] = {
            "last_date": str(last),
            "age_days": (today - last).days,
            "stale": (today - last).days > 5,
            "max_abs_daily_move": float(r.max()),
            "moves_over_25pct": int((r > 0.25).sum()),
            "zero_volume_days": int((f["volume"] == 0).sum()),
        }
    return out


def drift_report(
    reference: pd.DataFrame, recent: pd.DataFrame, ref_prob: np.ndarray, new_prob: np.ndarray
) -> dict[str, Any]:
    feats = {f: psi(reference[f].to_numpy(), recent[f].to_numpy()) for f in FEATURES}
    pp = psi(ref_prob, new_prob)
    return {
        "n_reference": int(len(reference)),
        "n_recent": int(len(recent)),
        "feature_psi": {f: {"psi": v, "level": _level(v)} for f, v in feats.items()},
        "prediction_psi": {"psi": pp, "level": _level(pp)},
        "prediction_mean": {"reference": float(ref_prob.mean()), "recent": float(new_prob.mean())},
        "thresholds": {"warn": PSI_WARN, "alert": PSI_ALERT},
    }

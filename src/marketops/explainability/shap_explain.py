"""SHAP explanations for the selected model (linear or tree, via the sklearn Pipeline)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from marketops.features.build import FEATURES


def _prepare(model: Any, x: pd.DataFrame) -> tuple[Any, np.ndarray]:
    """Return (final estimator, transformed matrix) so SHAP sees what the estimator sees."""
    steps = list(model.named_steps.items())
    z = x[FEATURES]
    for _, step in steps[:-1]:
        z = step.transform(z)
    return steps[-1][1], np.asarray(z, dtype=float)


def explain(model: Any, background: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """SHAP values (log-odds contribution of each feature) for `rows`, same index."""
    est, bg = _prepare(model, background.iloc[:100])  # shap caps linear backgrounds at 100
    _, z = _prepare(model, rows)
    if hasattr(est, "coef_"):
        explainer: Any = shap.LinearExplainer(est, bg)
    else:
        explainer = shap.TreeExplainer(est)
    vals = explainer.shap_values(z)
    return pd.DataFrame(np.asarray(vals), columns=FEATURES, index=rows.index)


def global_importance(model: Any, background: pd.DataFrame, rows: pd.DataFrame) -> dict[str, float]:
    sv = explain(model, background, rows)
    imp = sv.abs().mean().sort_values(ascending=False)
    return {str(k): float(v) for k, v in imp.items()}

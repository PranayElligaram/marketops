"""Load the latest trained model + stored evaluation artifacts and score symbols."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

from marketops.data.store import BENCHMARK, load_symbol
from marketops.features.build import FEATURES, symbol_features


class ModelNotAvailable(Exception):
    pass


class UnsupportedSymbol(Exception):
    pass


@dataclass
class LoadedModel:
    version: str
    directory: Path
    model: Any
    meta: dict[str, Any]


def load_latest(model_root: Path) -> LoadedModel:
    latest = model_root / "LATEST"
    if not latest.exists():
        raise ModelNotAvailable("No trained model found. Run `python -m marketops.training.cli`.")
    version = latest.read_text().strip()
    if not version or Path(version).name != version:
        raise ModelNotAvailable("LATEST does not contain a valid model version.")
    d = model_root / version
    try:
        meta = json.loads((d / "metadata.json").read_text())
        raw = (d / "model.joblib").read_bytes()
    except (OSError, ValueError) as exc:
        raise ModelNotAvailable(f"Model {version} could not be loaded: {exc}") from None
    _verify_artifact(version, meta, raw)
    try:
        model = joblib.load(io.BytesIO(raw))  # trusted local artifact only
    except Exception as exc:  # unpickling can fail in many ways (version skew, corruption)
        raise ModelNotAvailable(f"Model {version} could not be deserialised: {exc}") from None
    _verify_schema(version, meta, model)
    return LoadedModel(version, d, model, meta)


def _verify_artifact(version: str, meta: dict[str, Any], raw: bytes) -> None:
    """Metadata must describe this exact artefact (version directory and content hash)."""
    if meta.get("version") != version:
        raise ModelNotAvailable(
            f"LATEST points to {version} but its metadata describes {meta.get('version')}."
        )
    expected = meta.get("model_sha256")
    if not expected:
        raise ModelNotAvailable(f"Model {version} has no recorded hash; retrain to regenerate.")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ModelNotAvailable(f"Model {version} does not match its recorded SHA-256.")


def _verify_schema(version: str, meta: dict[str, Any], model: Any) -> None:
    """The feature schema the model was trained on must equal the code's current schema."""
    if list(meta.get("features", [])) != FEATURES:
        raise ModelNotAvailable(f"Model {version} was trained on a different feature schema.")
    n_in = getattr(model, "n_features_in_", len(FEATURES))
    if n_in != len(FEATURES):
        raise ModelNotAvailable(f"Model {version} expects {n_in} features, not {len(FEATURES)}.")
    trained = meta.get("env", {}).get("sklearn")
    if trained and trained != sklearn.__version__:
        raise ModelNotAvailable(
            f"Model {version} was trained with scikit-learn {trained}, runtime has "
            f"{sklearn.__version__}; retrain or align the environment."
        )


def load_backtest(lm: LoadedModel) -> dict[str, Any]:
    p = lm.directory / "backtest.json"
    if not p.exists():
        raise ModelNotAvailable("No backtest results for this model. Run marketops.backtest.run.")
    result: dict[str, Any] = json.loads(p.read_text())
    return result


def check_symbol(lm: LoadedModel, symbol: str) -> str:
    s = symbol.upper()
    if s not in lm.meta["symbols"] and s != BENCHMARK:
        raise UnsupportedSymbol(f"{s} is outside the trained universe {lm.meta['symbols']}")
    return s


def history(data_root: Path, lm: LoadedModel, symbol: str, days: int) -> pd.DataFrame:
    s = check_symbol(lm, symbol)
    return load_symbol(data_root, s).tail(days)


def analyse(data_root: Path, lm: LoadedModel, symbol: str) -> dict[str, Any]:
    s = check_symbol(lm, symbol)
    if s == BENCHMARK:
        raise UnsupportedSymbol("SPY is the benchmark and is not scored.")
    feats = symbol_features(load_symbol(data_root, s), load_symbol(data_root, BENCHMARK))
    feats = feats.dropna(subset=FEATURES)
    row = feats.iloc[[-1]]
    prob = float(lm.model.predict_proba(row[FEATURES])[0, 1])
    thr = float(lm.meta["threshold"])
    as_of = row["date"].iloc[0]
    stale_days = (date.today() - as_of.date()).days
    return {
        "symbol": s,
        "as_of_date": str(as_of.date()),
        "model_version": lm.version,
        "probability_up": prob,
        "threshold": thr,
        "signal": "above_threshold" if prob >= thr else "below_threshold",
        "features": {f: float(row[f].iloc[0]) for f in FEATURES},
        "data_age_days": stale_days,
        "warnings": (["Market data is more than 5 days old."] if stale_days > 5 else []),
    }

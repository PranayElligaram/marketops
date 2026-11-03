"""Read-only research endpoints: market data, ML signal, evaluation, backtests.

Nothing here places orders. Responses carry provenance and a research disclaimer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from marketops.config import get_settings
from marketops.data.store import load_symbol
from marketops.inference import service
from marketops.monitoring.checks import data_quality
from marketops.persistence import database, models

router = APIRouter(tags=["research"])
logger = logging.getLogger("marketops.api.research")

DISCLAIMER = (
    "Research output, not investment advice. Held-out evidence shows no demonstrated "
    "predictive edge; do not trade on this signal."
)


class Analysis(BaseModel):
    symbol: str
    as_of_date: str
    model_version: str
    probability_up: float = Field(ge=0, le=1)
    threshold: float
    signal: str
    features: dict[str, float]
    data_age_days: int
    warnings: list[str]
    persisted: bool
    disclaimer: str = DISCLAIMER


class Bar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float


class History(BaseModel):
    symbol: str
    provider: str
    rows: list[Bar]
    price_convention: str = "close=split-adjusted; adj_close=split+dividend adjusted"
    terms: str = "Personal research use only; not redistributable."


class PredictionPoint(BaseModel):
    date: str
    probability_up: float
    realised_label: int
    split: str


def _model() -> service.LoadedModel:
    try:
        return service.load_latest(Path(get_settings().model_dir))
    except service.ModelNotAvailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from None


def _guard(fn: Any, *a: Any) -> Any:
    try:
        return fn(*a)
    except service.UnsupportedSymbol as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from None
    except service.ModelNotAvailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from None
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Cached data invalid: {exc}") from None


@router.get("/model/metadata")
def model_metadata() -> dict[str, Any]:
    lm = _model()
    try:
        models.register_model(database.get_engine(), lm.meta)
    except Exception as exc:  # DB optional for reads
        logger.warning("model registry write skipped: %s", type(exc).__name__)
    keys = (
        "version",
        "target",
        "selected_model",
        "features",
        "threshold",
        "dev_end",
        "test_start",
        "symbols",
        "latency",
        "data",
        "env",
        "caveats",
        "train_rows_final",
        "test_rows",
    )
    return {k: lm.meta.get(k) for k in keys}


@router.get("/model/evaluation")
def model_evaluation() -> dict[str, Any]:
    lm = _model()
    return {
        k: lm.meta.get(k)
        for k in (
            "version",
            "walk_forward_cv",
            "dev_baselines",
            "test",
            "test_other_candidates_not_for_selection",
            "caveats",
        )
    }


@router.get("/model/explain")
def model_explain() -> dict[str, Any]:
    """Global SHAP importance (mean |log-odds contribution|, held-out sample)."""
    lm = _model()
    p = lm.directory / "shap.json"
    if not p.exists():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "No SHAP report for this model.")
    return {"version": lm.version, **json.loads(p.read_text())}


@router.get("/monitoring")
def monitoring() -> dict[str, Any]:
    """Live data-quality checks plus the drift report written at training time."""
    lm = _model()
    root = Path(get_settings().data_dir)
    frames = {s: _guard(load_symbol, root, s) for s in [*lm.meta["symbols"], "SPY"]}
    out: dict[str, Any] = {"version": lm.version, "data_quality": data_quality(frames)}
    p = lm.directory / "monitoring.json"
    out["drift"] = json.loads(p.read_text()) if p.exists() else None
    return out


@router.get("/backtest")
def backtest_results() -> dict[str, Any]:
    return _guard(lambda: service.load_backtest(_model()))  # type: ignore[no-any-return]


@router.get("/market/{symbol}/history", response_model=History)
def market_history(symbol: str, days: int = Query(250, ge=5, le=6000)) -> History:
    lm = _model()
    df: pd.DataFrame = _guard(service.history, Path(get_settings().data_dir), lm, symbol, days)
    rows = [
        Bar(**{**{str(k): v for k, v in rec.items()}, "date": str(rec["date"].date())})
        for rec in df[["date", "open", "high", "low", "close", "adj_close", "volume"]].to_dict(
            "records"
        )
    ]
    return History(symbol=symbol.upper(), provider="yahoo_chart", rows=rows)


@router.get("/analysis/{symbol}", response_model=Analysis)
def analysis(symbol: str) -> Analysis:
    lm = _model()
    result = _guard(service.analyse, Path(get_settings().data_dir), lm, symbol)
    persisted = False
    try:
        engine = database.get_engine()
        models.init_schema(engine)
        persisted = models.log_prediction(
            engine,
            {
                "model_version": result["model_version"],
                "symbol": result["symbol"],
                "as_of_date": result["as_of_date"],
                "probability_up": result["probability_up"],
                "threshold": result["threshold"],
                "signal": result["signal"],
                "features": result["features"],
            },
        )
    except Exception as exc:
        logger.warning("analysis persistence skipped: %s", type(exc).__name__)
    return Analysis(**result, persisted=persisted)


@router.get("/analysis", response_model=None)
def analysis_history(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
    try:
        engine = database.get_engine()
        models.init_schema(engine)
        return models.recent_predictions(engine, limit)
    except Exception:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database unavailable.") from None


@router.get("/predictions/{symbol}", response_model=list[PredictionPoint])
def prediction_history(
    symbol: str, split: str = Query("test", pattern="^(test|dev_oof)$")
) -> list[PredictionPoint]:
    """Stored out-of-sample probabilities (dev = walk-forward OOF, test = held-out)."""
    lm = _model()
    s = _guard(service.check_symbol, lm, symbol)
    path = lm.directory / "predictions.csv"
    if not path.exists():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "No stored predictions.")
    p = pd.read_csv(path, parse_dates=["date"])
    p = p[(p["symbol"] == s) & (p["split"] == split)]
    return [
        PredictionPoint(
            date=str(r.date.date()),
            probability_up=r.prob,
            realised_label=int(r.label),
            split=r.split,
        )
        for r in p.itertuples()
    ]

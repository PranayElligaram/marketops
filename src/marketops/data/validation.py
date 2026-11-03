"""Strict validation of daily OHLCV frames. Fails loudly; does not repair silently."""

from __future__ import annotations

import pandas as pd

COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume", "dividend", "split"]
PRICE_COLS = ["open", "high", "low", "close", "adj_close"]


class DataQualityError(ValueError):
    pass


def clean_and_validate(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Drop provider placeholder rows (all-null prices), then enforce: schema, no remaining
    nulls, no duplicate dates, chronological order, positive prices, non-negative volume,
    OHLC consistency, and valid corporate-action values."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise DataQualityError(f"{symbol}: missing columns {missing}")
    out = df[COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.dropna(subset=PRICE_COLS, how="all")
    if out.empty:
        raise DataQualityError(f"{symbol}: no usable rows")
    null_cols = out.columns[out.isna().any()].tolist()
    if null_cols:
        raise DataQualityError(f"{symbol}: null values in {null_cols}")
    for c in COLUMNS[1:]:
        out[c] = out[c].astype(float)
    if out["date"].duplicated().any():
        raise DataQualityError(f"{symbol}: duplicate dates")
    if not out["date"].is_monotonic_increasing:
        raise DataQualityError(f"{symbol}: dates not in chronological order")
    if (out[PRICE_COLS] <= 0).any().any():
        raise DataQualityError(f"{symbol}: non-positive price")
    if (out["volume"] < 0).any():
        raise DataQualityError(f"{symbol}: negative volume")
    tol = 1e-6
    bad = (out["high"] < out[["open", "close", "low"]].max(axis=1) - tol) | (
        out["low"] > out[["open", "close", "high"]].min(axis=1) + tol
    )
    if bad.any():
        raise DataQualityError(f"{symbol}: {int(bad.sum())} rows violate OHLC consistency")
    if (out["dividend"] < 0).any() or (out["split"] <= 0).any():
        raise DataQualityError(f"{symbol}: invalid corporate-action values")
    return out.reset_index(drop=True)

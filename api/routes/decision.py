"""Deterministic read-only technical research, with no broker or trade operations."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException

from api.routes.research import market_history
from marketops.decision.technical import DailyBar, screen_daily_bars

router = APIRouter(tags=["research"])


@router.get("/research/technical/{symbol}")
def technical_screen(symbol: str) -> dict[str, Any]:
    """Review cached daily prices for supported symbols using existing market-data validation."""
    history = market_history(symbol, days=300)
    bars = [
        DailyBar(
            day=datetime.strptime(bar.date, "%Y-%m-%d").date(),
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
        )
        for bar in history.rows
    ]
    try:
        result = screen_daily_bars(bars, as_of=datetime.now(UTC).date())
    except ValueError:
        raise HTTPException(502, "Cached historical OHLCV is invalid.") from None
    return {
        "symbol": history.symbol,
        "provider": history.provider,
        "price_convention": history.price_convention,
        "terms": history.terms,
        **asdict(result),
        "as_of": result.as_of.isoformat() if result.as_of is not None else None,
    }

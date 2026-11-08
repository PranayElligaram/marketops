"""Read-only SEC EDGAR filing metadata for a ticker. No LLM, no scraping, no broker access."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query

from marketops.config import get_settings
from marketops.data.sec_filings import SecClient, SecUnavailable

router = APIRouter(tags=["research"])
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


@lru_cache(maxsize=4)
def _client(user_agent: str) -> SecClient:
    return SecClient(user_agent)


@router.get("/research/filings/{symbol}")
def recent_filings(symbol: str, limit: Annotated[int, Query(ge=1, le=30)] = 15) -> dict[str, Any]:
    """Recent 8-K/10-K/10-Q-type filings with accession numbers, links and retrieval time."""
    ticker = symbol.strip().upper()
    if not _SYMBOL.fullmatch(ticker):
        raise HTTPException(422, "Invalid ticker.")
    user_agent = get_settings().sec_user_agent.strip()
    if len(user_agent) < 8 or "@" not in user_agent:
        raise HTTPException(
            503,
            "SEC filings are not configured: set MARKETOPS_SEC_USER_AGENT to a descriptive "
            "name and contact email, as SEC EDGAR fair-access rules require.",
        )
    try:
        return _client(user_agent).recent_filings(ticker, limit)
    except LookupError:
        raise HTTPException(404, "Ticker not found in the SEC company list.") from None
    except SecUnavailable as exc:
        raise HTTPException(502, str(exc)) from None

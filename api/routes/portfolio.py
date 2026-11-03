"""Portfolio API with explicitly selected real or synthetic data source."""

import threading
import time

from fastapi import APIRouter, Depends, HTTPException, status

from api.security import private_access
from marketops.broker.t212 import (
    BrokerError,
    Trading212ReadOnlyClient,
    require_environment_allowed,
)
from marketops.config import get_settings
from marketops.portfolio.analytics import (
    EXAMPLE_ACCOUNT_SUMMARY,
    EXAMPLE_POSITIONS,
    PortfolioOverview,
    summarise_positions,
)

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

# Trading 212 limits: positions 1 req/s, account summary 1 req/5 s. Serving a snapshot younger
# than this avoids 429s from dashboard refreshes. The overview carries snapshot_utc.
CACHE_TTL_S = 10.0
_cache_lock = threading.Lock()
_cache: tuple[float, str, PortfolioOverview] | None = None


def clear_cache() -> None:
    global _cache
    with _cache_lock:
        _cache = None


@router.get("/example", response_model=PortfolioOverview)
def example() -> PortfolioOverview:
    return summarise_positions(
        EXAMPLE_POSITIONS, source="synthetic_example", summary=EXAMPLE_ACCOUNT_SUMMARY
    )


@router.get("/overview", response_model=PortfolioOverview, dependencies=[Depends(private_access)])
def overview() -> PortfolioOverview:
    global _cache
    settings = get_settings()
    if not settings.broker_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Trading 212 read-only keys are not configured locally. Use /portfolio/example."
            ),
        )
    env = settings.t212_environment
    with _cache_lock:
        if _cache and _cache[1] == env and time.monotonic() - _cache[0] < CACHE_TTL_S:
            return _cache[2]
    try:
        require_environment_allowed(env, settings.t212_allow_live)
        client = Trading212ReadOnlyClient(
            api_key=settings.t212_api_key,
            api_secret=settings.t212_api_secret,
            environment=env,
        )
        positions = client.get_positions()
        try:
            summary = client.get_account_summary()
        except BrokerError:
            summary = None  # cash is optional; positions remain useful
        result = summarise_positions(positions, source="trading212", summary=summary)
        result.broker_environment = env
    except BrokerError as exc:
        code = (
            status.HTTP_429_TOO_MANY_REQUESTS
            if "rate limit" in str(exc)
            else status.HTTP_403_FORBIDDEN
            if "disabled" in str(exc)
            else status.HTTP_502_BAD_GATEWAY
        )
        raise HTTPException(status_code=code, detail=str(exc)) from None
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Broker returned incompatible financial data. No portfolio totals were calculated."
            ),
        ) from None
    with _cache_lock:
        _cache = (time.monotonic(), env, result)
    return result

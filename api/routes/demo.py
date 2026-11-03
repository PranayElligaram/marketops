"""Public synthetic demonstration of the historical valuation and benchmark comparison.

Returns only invented data generated in-process: no credentials, database, broker or network.
"""

from fastapi import APIRouter

from marketops.demo import build_demo
from marketops.valuation.report import HistoricalValuation

router = APIRouter(prefix="/performance", tags=["demo"])

_cache: HistoricalValuation | None = None


@router.get("/demo", response_model=HistoricalValuation)
def demo() -> HistoricalValuation:
    global _cache
    if _cache is None:
        _cache = build_demo()
    return _cache

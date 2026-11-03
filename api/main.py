"""MarketOps FastAPI entrypoint."""

import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from api.routes.agent import router as agent_router
from api.routes.health import router as health_router
from api.routes.history import router as history_router
from api.routes.performance import router as performance_router
from api.routes.portfolio import router as portfolio_router
from api.routes.research import router as research_router
from marketops import __version__
from marketops.config import get_settings
from marketops.logging_config import configure_logging

configure_logging(get_settings().log_level)
logger = logging.getLogger("marketops.api")

app = FastAPI(
    title="MarketOps API",
    version=__version__,
    description="Local-first stock research, ML signals and read-only portfolio analytics.",
)
app.include_router(health_router)
app.include_router(portfolio_router)
app.include_router(history_router)
app.include_router(performance_router)
app.include_router(research_router)
app.include_router(agent_router)


@app.middleware("http")
async def request_telemetry(request: Request, call_next: RequestResponseEndpoint) -> Response:
    started = time.perf_counter()
    request_id = str(uuid4())
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled API request failure")
        raise
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_completed method=%s path=%s status=%s latency_ms=%s request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )
    return response


@app.get("/", tags=["meta"])
def index() -> dict[str, str]:
    return {"service": "marketops-api", "phase": "research", "version": __version__}

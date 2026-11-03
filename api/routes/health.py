"""Liveness and dependency-readiness endpoints."""

from typing import Literal

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from marketops.persistence import database

router = APIRouter(tags=["health"])


class Liveness(BaseModel):
    status: Literal["ok"] = "ok"
    service: Literal["marketops-api"] = "marketops-api"


@router.get("/health/live", response_model=Liveness)
def live() -> Liveness:
    """Process liveness; deliberately does not touch the DB."""
    return Liveness()


@router.get("/health/ready", response_model=None)
def ready() -> JSONResponse:
    """Return 503 until PostgreSQL responds to SELECT 1."""
    is_ready = database.database_is_ready()
    return JSONResponse(
        status_code=status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "ready" if is_ready else "unavailable",
            "dependencies": {"postgres": "up" if is_ready else "down"},
        },
    )

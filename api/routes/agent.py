"""Bounded research agent endpoints. Read-only; the LLM is a local Ollama model."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Query, status

from api.security import require_private_token
from marketops.agents.ollama import OllamaClient
from marketops.agents.runner import run_research
from marketops.agents.schemas import AgentRequest, Report
from marketops.config import get_settings
from marketops.persistence import database, models

router = APIRouter(prefix="/agent", tags=["agent"])
logger = logging.getLogger("marketops.api.agent")


def get_client() -> OllamaClient | None:
    """Overridable in tests. None means the configured URL was refused."""
    s = get_settings()
    try:
        return OllamaClient(s.ollama_base_url, s.ollama_model, timeout=s.agent_timeout_s)
    except ValueError:
        return None


_require_token = require_private_token


@router.get("/status")
def agent_status() -> dict[str, Any]:
    s = get_settings()
    client = get_client()
    llm = (
        client.status()
        if client
        else {"ollama": "refused_url", "model": s.ollama_model, "model_installed": False}
    )
    return {
        "llm": llm,
        "limits": {
            "max_iterations": s.agent_max_iterations,
            "max_tool_calls": s.agent_max_tool_calls,
            "timeout_s": s.agent_timeout_s,
        },
        "broker_portfolio_enabled": s.agent_allow_broker_portfolio and s.broker_configured,
    }


@router.post("/research", response_model=Report)
def research(
    req: AgentRequest,
    x_marketops_token: Annotated[str | None, Header()] = None,
) -> Report:
    s = get_settings()
    if req.portfolio_source == "trading212" and not (
        s.agent_allow_broker_portfolio and s.broker_configured
    ):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Broker portfolio access for the agent is disabled or unconfigured. "
            "Use portfolio_source='example' for demonstration data.",
        )
    if req.portfolio_source == "trading212":
        _require_token(x_marketops_token)
    report, audit = run_research(req.question, req.portfolio_source, s, get_client())
    try:
        models.log_agent_run(database.get_engine(), audit)
    except Exception as exc:  # DB optional; audit is best-effort
        logger.warning("agent audit skipped: %s", type(exc).__name__)
    return report


@router.get("/runs", response_model=None)
def runs(
    limit: int = Query(20, ge=1, le=100),
    x_marketops_token: Annotated[str | None, Header()] = None,
) -> list[dict[str, Any]]:
    s = get_settings()
    if s.agent_allow_broker_portfolio:  # audit may then describe broker-backed runs
        _require_token(x_marketops_token)
    try:
        return models.recent_agent_runs(database.get_engine(), limit)
    except Exception:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database unavailable.") from None

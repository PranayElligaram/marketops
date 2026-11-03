"""Typed contracts for the research agent report and its audit trail."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DISCLAIMER = (
    "Research output, not investment advice. MarketOps cannot place orders. The ML model has no "
    "demonstrated predictive edge; do not trade on its output."
)

PortfolioSource = Literal["none", "example", "trading212"]


class Evidence(BaseModel):
    id: str
    tool: str
    args: dict[str, Any]
    kind: Literal["fact", "model"]
    source: str
    retrieved_utc: str
    sha256: str  # "withheld" for sensitive evidence (a hash of low-entropy data aids guessing)
    sensitive: bool = False
    data: dict[str, Any] | None = None  # withheld for sensitive (broker) evidence


class Claim(BaseModel):
    text: str
    evidence: list[str] = Field(default_factory=list)


class RejectedClaim(BaseModel):
    section: str
    text: str
    reason: str


class ToolTrace(BaseModel):
    tool: str
    args: dict[str, Any]
    status: Literal["ok", "error", "rejected", "limit"]
    error: str | None = None
    duration_ms: float = 0.0
    evidence_id: str | None = None
    origin: Literal["llm", "system", "fallback"] = "llm"


class Timing(BaseModel):
    """Where one run spent its time (ms) and how much text went through the model."""

    llm_tool_phase_ms: float = 0.0
    llm_synthesis_ms: float = 0.0
    tools_ms: float = 0.0
    verify_ms: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0


class Report(BaseModel):
    run_id: str
    status: Literal["completed", "completed_with_warnings", "failed"]
    mode: Literal["llm", "deterministic_fallback"]
    llm_model: str
    llm_status: str
    question: str
    symbols: list[str]
    portfolio_source: PortfolioSource
    verified_facts: list[Claim]
    model_estimates: list[Claim]
    interpretation: list[Claim]
    missing_or_uncertain: list[str]
    rejected_claims: list[RejectedClaim]
    limitations: list[str]
    warnings: list[str]
    evidence: list[Evidence]
    tool_trace: list[ToolTrace]
    iterations: int
    tool_calls: int
    duration_ms: float
    timing: Timing = Field(default_factory=Timing)
    disclaimer: str = DISCLAIMER


class AgentRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    portfolio_source: PortfolioSource = "none"

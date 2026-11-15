"""Structured inputs/outputs shared by tools, orchestrator and API."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

ToolStatus = Literal["ok", "error", "timeout", "invalid_request", "unavailable", "denied", "confirmation_required"]


class ToolResult(BaseModel):
    """Uniform envelope returned by every tool. `data` is evidence; the agent may only
    state operational facts that appear here."""

    tool: str
    status: ToolStatus
    data: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    request_id: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    confirm_actions: bool = Field(
        default=False,
        description="Explicit user confirmation for state-changing operations (Airflow trigger, K8s restart).",
    )


class Evidence(BaseModel):
    tool: str
    status: str
    summary: Any = None


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    tools_called: list[str]
    planner: str = "rules"
    evidence: list[Evidence]
    known_facts: list[str] = []
    uncertainties: list[str] = []
    recommended_actions: list[str] = []


class PredictRequest(BaseModel):
    features: list[float]


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str

# © Artur Czarnecki. All rights reserved.

"""Plane A orchestration summary (architecture §31.1 · ACP-OBS-2)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.agent_run_enums import AgentRunStatus


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AgentInvocationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str
    run_id: str
    status: AgentRunStatus
    step_count: int = 0
    total_llm_tokens: int = 0
    total_tool_calls: int = 0
    terminal_reason: str | None = None


class ApplicationRunSummary(BaseModel):
    """Host-facing summary for multi-agent orchestration (Plane A)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["application_run_summary.v1"] = "application_run_summary.v1"
    task_id: str
    graph_id: str
    terminal_status: AgentRunStatus
    agent_invocations: list[AgentInvocationSummary] = Field(default_factory=list)
    total_agents: int = 0
    total_steps: int = 0
    total_llm_tokens: int = 0
    completed_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

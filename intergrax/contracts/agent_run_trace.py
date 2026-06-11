# © Artur Czarnecki. All rights reserved.

"""Plane B agent execution journal (architecture §31.2 · ACP-OBS-1)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.agent_run_enums import (
    AgentRunErrorCode,
    StepNextAction,
    TerminalReason,
)
from intergrax.contracts.runtime_policy import PolicyAction


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AgentStepStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    PAUSED = "paused"


class GatewayCallStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"


class PolicyCheckPhase(StrEnum):
    PRE = "pre"
    POST = "post"


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str
    tool_id: str
    status: GatewayCallStatus
    latency_ms: int = 0
    args_digest: str = ""
    error_code: str | None = None
    policy_rule_id: str | None = None


class RagCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str
    collection_id: str
    status: GatewayCallStatus
    latency_ms: int = 0
    hit_count: int = 0
    policy_rule_id: str | None = None


class LlmCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str
    model_id: str
    provider: str
    status: GatewayCallStatus
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    model_hint: str | None = None
    policy_rule_id: str | None = None


class PolicyVerdictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: PolicyCheckPhase
    action: PolicyAction
    reason: str
    policy_rule_id: str


class AgentStepRecord(BaseModel):
    """One step in the agent execution journal (Plane B)."""

    model_config = ConfigDict(extra="forbid")

    step_id: str
    step_index: int
    started_at: datetime = Field(default_factory=_utc_now)
    finished_at: datetime | None = None
    status: AgentStepStatus
    next_action: StepNextAction
    terminal_reason: TerminalReason | None = None
    state_version: int
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    rag_calls: list[RagCallRecord] = Field(default_factory=list)
    llm_calls: list[LlmCallRecord] = Field(default_factory=list)
    policy_verdicts: list[PolicyVerdictRecord] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    error_code: AgentRunErrorCode | None = None


class AgentRunTrace(BaseModel):
    """Full Plane B trace for one agent run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["agent_run_trace.v1"] = "agent_run_trace.v1"
    run_id: str = ""
    steps: list[AgentStepRecord] = Field(default_factory=list)
    total_steps: int = 0
    total_llm_tokens: int = 0
    total_tool_calls: int = 0
    total_rag_calls: int = 0

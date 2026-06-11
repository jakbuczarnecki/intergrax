# © Artur Czarnecki. All rights reserved.

"""Typed agent run I/O contracts (architecture §29 · ACP-DX-1 · ACP-CON-1)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.agent_run_enums import (
    AgentRunAutonomyLevel,
    AgentRunErrorCode,
    AgentRunStatus,
    PrincipalType,
    SideEffectMode,
    TerminalReason,
)
from intergrax.contracts.agent_run_trace import AgentRunTrace
from intergrax.contracts.artifact_ref import ArtifactRef
from intergrax.contracts.memory_scope import MemoryScope


class RequestIdentity(BaseModel):
    """Authenticated principal for a run (architecture §30.9)."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    user_id: str | None = None
    principal_type: PrincipalType = PrincipalType.USER
    auth_subject: str | None = None

    @field_validator("tenant_id")
    @classmethod
    def _tenant_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tenant_id must be non-empty")
        return value


class AgentEnvironmentOverrides(BaseModel):
    """Per-run narrow of host profile slices (architecture §30.3)."""

    model_config = ConfigDict(extra="forbid")

    tool_allowlist_add: list[str] = Field(default_factory=list)
    tool_allowlist_remove: list[str] = Field(default_factory=list)
    skill_ids_override: list[str] = Field(default_factory=list)
    memory_namespace: str | None = None
    memory_scope: MemoryScope | None = None
    rag_collection: str | None = None
    rag_collection_ids: list[str] = Field(default_factory=list)
    llm_profile_id: str | None = None
    llm_profile_slug: str | None = None
    metadata_patch: dict[str, Any] = Field(default_factory=dict)


class AgentExecutionOptions(BaseModel):
    """Policy-bound execution hints for one agent session."""

    model_config = ConfigDict(extra="forbid")

    max_steps: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, ge=0.0)
    max_wall_ms: int | None = Field(default=None, ge=1)
    autonomy_level: AgentRunAutonomyLevel = AgentRunAutonomyLevel.BALANCED
    side_effect_mode: SideEffectMode = SideEffectMode.IMMEDIATE
    checkpoint_every_step: bool = True


class AgentRunError(BaseModel):
    """Structured run/step error (architecture §37.4)."""

    model_config = ConfigDict(extra="forbid")

    code: AgentRunErrorCode
    message: str
    step_index: int | None = None
    retriable: bool = False
    details: dict[str, Any] | None = None


class ComplianceSummary(BaseModel):
    """Org + platform policy rollup on AgentRunResult (architecture §39.5 · ACP-ORG-4)."""

    model_config = ConfigDict(extra="forbid")

    deny_count: int = 0
    warn_count: int = 0
    rules_triggered: list[str] = Field(default_factory=list)


class AgentRunCost(BaseModel):
    """Cost rollup attached to AgentRunResult."""

    model_config = ConfigDict(extra="forbid")

    tokens_in: int = 0
    tokens_out: int = 0
    llm_usd: float = 0.0
    tool_units: int = 0
    total_usd: float = 0.0


class GovernanceSnapshot(BaseModel):
    """HITL / interrupt resolution metadata when paused."""

    model_config = ConfigDict(extra="forbid")

    hitl_ticket_id: str | None = None
    pause_cause: str | None = None
    approver: str | None = None
    resume_token: str | None = None


class AgentRunRequest(BaseModel):
    """Public entry contract for Agent.run() (architecture §29.2)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["agent_run.v1"] = "agent_run.v1"
    input: str | dict[str, Any]
    identity: RequestIdentity
    session_id: str | None = None
    correlation_id: str | None = None
    agent_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    state: dict[str, Any] | None = None
    environment_overrides: AgentEnvironmentOverrides | None = None
    execution_options: AgentExecutionOptions | None = None


class AgentRunResult(BaseModel):
    """Typed result from Agent.run() (architecture §29.2)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["agent_run.v1"] = "agent_run.v1"
    status: AgentRunStatus
    output: str | dict[str, Any] = ""
    state: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    structured_data: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    errors: list[AgentRunError] = Field(default_factory=list)
    warnings: list[AgentRunError] = Field(default_factory=list)
    trace_id: str = ""
    run_id: str = ""
    trace: AgentRunTrace = Field(default_factory=AgentRunTrace)
    used_tools: list[str] = Field(default_factory=list)
    cost: AgentRunCost | None = None
    duration_ms: int = 0
    terminal_reason: TerminalReason | None = None
    governance: GovernanceSnapshot | None = None
    compliance_summary: ComplianceSummary | None = None

    @model_validator(mode="after")
    def _terminal_reason_required_for_terminal_status(self) -> AgentRunResult:
        if self.status in {
            AgentRunStatus.SUCCEEDED,
            AgentRunStatus.FAILED,
            AgentRunStatus.PAUSED,
            AgentRunStatus.CANCELLED,
        } and self.terminal_reason is None:
            raise ValueError("terminal_reason is required for terminal or paused status")
        return self

    @model_validator(mode="after")
    def _failed_status_requires_errors(self) -> AgentRunResult:
        if self.status == AgentRunStatus.FAILED and not self.errors:
            raise ValueError("errors must be non-empty when status is failed")
        return self


def require_user_id_for_user_memory_scope(
    identity: RequestIdentity,
    *,
    memory_scope: Literal["user", "org"],
) -> None:
    """§30.9 gate — user-scoped memory requires authenticated user_id."""
    if memory_scope == "user" and not (identity.user_id or "").strip():
        raise ValueError(
            "VALIDATION_FAILED: user_id is required when memory_scope is user"
        )

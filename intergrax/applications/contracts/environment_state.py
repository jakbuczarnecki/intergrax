# © Artur Czarnecki. All rights reserved.

"""Typed application environment state for ApplicationHost hooks (APP-CON-2)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from intergrax.applications.contracts.execution_mode import ExecutionMode


class EnvironmentTaskPhase(StrEnum):
    """High-level Nexus task phase for host-visible state."""

    INTAKE = "intake"
    CLASSIFICATION = "classification"
    PLANNING = "planning"
    GRAPH_EXECUTION = "graph_execution"
    AGENT_SELECTION = "agent_selection"
    AGENT_RUN = "agent_run"
    HITL = "hitl"
    FINALIZATION = "finalization"
    COMPLETED = "completed"
    FAILED = "failed"


class EnvironmentHealthStatus(StrEnum):
    """Coarse environment health for hooks and ops dashboards."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    BUDGET_PRESSURE = "budget_pressure"
    POLICY_BLOCKED = "policy_blocked"
    HITL_PENDING = "hitl_pending"
    FAILED = "failed"


class HitlEscalationState(BaseModel):
    """HITL / escalation snapshot for application hooks."""

    model_config = ConfigDict(extra="forbid")

    pending: bool = False
    ticket_id: str | None = None
    escalation_reason: str | None = None
    awaiting_role: str | None = None


class ActiveBudgetState(BaseModel):
    """Budget metering + limit posture visible to application hooks."""

    model_config = ConfigDict(extra="forbid")

    agent_tokens_total: int = 0
    environment_tokens_total: int = 0
    agent_tokens_limit: int | None = None
    environment_tokens_limit: int | None = None
    warn_threshold_ratio: float = 0.80
    warn_emitted: bool = False
    hard_exceeded: bool = False
    last_reaction: str | None = None


class WorkspaceIsolationRef(BaseModel):
    """Active shadow workspace handle for a task."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    tenant_id: str
    task_id: str
    root_path: str | None = None


class SandboxIsolationRef(BaseModel):
    """Active sandbox session handle for a task."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    tenant_id: str
    task_id: str
    root_path: str | None = None


class PolicyOverlayState(BaseModel):
    """Active org/policy overlays merged for this task."""

    model_config = ConfigDict(extra="forbid")

    organization_id: str | None = None
    org_role_id: str | None = None
    active_scenario_id: str | None = None
    playbook_ids: list[str] = Field(default_factory=list)
    prompt_overlay_ids: list[str] = Field(default_factory=list)
    effective_tool_denies: list[str] = Field(default_factory=list)


class PendingNotification(BaseModel):
    """Queued operator/user notification from host reactions."""

    model_config = ConfigDict(extra="forbid")

    channel: str
    template_id: str | None = None
    payload_ref: str | None = None


class ApplicationEnvironmentState(BaseModel):
    """
    Host-scoped state surfaced on ``HookContext.runtime_state`` for Tier-3 hooks.

    Wire format key: ``app_env_state.v1`` inside ``HookContext.runtime_state``.

    **Persistence rule:** task-scoped by default — survives hook invocations within
    one ``Task`` lifecycle via MODIFY merges. Cross-task persistence MUST use Tier-0
    stores (task memory, trace DB) — not unbounded growth in ``custom``.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "app_env_state.v2"
    app_id: str
    profile_id: str
    profile_snapshot_id: str | None = None
    execution_mode: ExecutionMode = ExecutionMode.BALANCED

    task_id: str | None = None
    run_id: str | None = None
    graph_id: str | None = None
    phase: EnvironmentTaskPhase = EnvironmentTaskPhase.INTAKE
    health: EnvironmentHealthStatus = EnvironmentHealthStatus.HEALTHY

    organization_id: str | None = None
    policy_overlays: PolicyOverlayState = Field(default_factory=PolicyOverlayState)

    hitl: HitlEscalationState = Field(default_factory=HitlEscalationState)
    budget: ActiveBudgetState = Field(default_factory=ActiveBudgetState)

    shadow_workspace: WorkspaceIsolationRef | None = None
    sandbox_session: SandboxIsolationRef | None = None

    pending_notifications: list[PendingNotification] = Field(default_factory=list)
    custom: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_runtime_state(cls, runtime_state: dict[str, Any]) -> ApplicationEnvironmentState | None:
        raw = runtime_state.get("app_env_state.v1")
        if raw is None:
            return None
        if isinstance(raw, ApplicationEnvironmentState):
            return raw
        if isinstance(raw, dict):
            return cls.model_validate(raw)
        return None

    def apply_to_runtime_state(self, runtime_state: dict[str, Any]) -> dict[str, Any]:
        merged = dict(runtime_state)
        merged["app_env_state.v1"] = self.model_dump(mode="json")
        return merged

    def patch_runtime_state(self) -> dict[str, Any]:
        """Payload for ``HookResult.modified_payload`` when updating host state."""
        return {"app_env_state.v1": self.model_dump(mode="json")}


def seed_application_environment_state(
    *,
    app_id: str,
    profile_id: str,
    execution_mode: ExecutionMode,
    task_id: str | None = None,
    organization_id: str | None = None,
    active_scenario_id: str | None = None,
    profile_snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Bootstrap ``HookContext.runtime_state`` for task intake hooks."""
    state = ApplicationEnvironmentState(
        app_id=app_id,
        profile_id=profile_id,
        profile_snapshot_id=profile_snapshot_id or profile_id,
        execution_mode=execution_mode,
        task_id=task_id,
        organization_id=organization_id,
        policy_overlays=PolicyOverlayState(
            organization_id=organization_id,
            active_scenario_id=active_scenario_id,
        ),
    )
    return state.apply_to_runtime_state({})

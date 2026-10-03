# © Artur Czarnecki. All rights reserved.

"""Cross-layer middleware hook semantics (ADR-CTRL-X-001, Tier-0)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.agent_contract_meta import AgentRiskLevel
from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.contracts.data_classification import DataClassification
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.middleware_hook_point import HookPoint
from intergrax.contracts.structured_json_value import JsonObject


class MiddlewareExecutionSubjectFacet(BaseModel):
    """Tenant and actor scope for middleware — absence stays explicit (no sentinels)."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str | None = None
    resource_tenant_id: str | None = None
    user_id: str | None = None


class EmptyMiddlewareHookPayload(BaseModel):
    """Applicable when no typed payload is required for the hook point."""

    model_config = ConfigDict(extra="forbid")


class LlmInferenceHookPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str | None = None
    llm_output: str | None = None


class ToolCallHookPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_id: str
    tool_name: str | None = None
    request_id: str | None = None
    arguments: JsonObject = Field(default_factory=dict)
    capability_ids: list[str] = Field(default_factory=list)
    allowed_tool_ids: list[str] = Field(default_factory=list)
    autonomy_level: AutonomyLevel | None = None
    agent_risk_level: AgentRiskLevel | None = None


class TaskIntakeHookPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability: str | None = None
    classification: str | None = None


class ContextBuildHookPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str | None = None
    capability: str | None = None


class DataProtectionRestrictedValue(BaseModel):
    """Structured memory/tool material subject to classification and encryption."""

    model_config = ConfigDict(extra="forbid")

    data_classification: DataClassification | None = None
    classification: str | None = None
    secret: str | None = None
    payload: str | None = None
    content: str | None = None
    data: str | None = None


class DataProtectionHookPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_classification: DataClassification | None = None
    classification: str | None = None
    namespace: str | None = None
    key: str | None = None
    write_policy: str | None = None
    value: DataProtectionRestrictedValue | None = None


type MiddlewareHookPayload = (
    EmptyMiddlewareHookPayload
    | LlmInferenceHookPayload
    | ToolCallHookPayload
    | TaskIntakeHookPayload
    | ContextBuildHookPayload
    | DataProtectionHookPayload
)


@runtime_checkable
class MiddlewareHookInvocationContext(Protocol):
    """Strongly typed cross-layer middleware invocation context."""

    task_id: str
    run_id: str
    node_id: str | None
    agent_id: str | None
    step_id: str | None
    phase: ExecutionPhase
    hook_point: HookPoint

    @property
    def payload(self) -> MiddlewareHookPayload: ...

    @property
    def subject(self) -> MiddlewareExecutionSubjectFacet: ...


__all__ = [
    "ContextBuildHookPayload",
    "DataProtectionHookPayload",
    "DataProtectionRestrictedValue",
    "EmptyMiddlewareHookPayload",
    "LlmInferenceHookPayload",
    "MiddlewareExecutionSubjectFacet",
    "MiddlewareHookInvocationContext",
    "MiddlewareHookPayload",
    "TaskIntakeHookPayload",
    "ToolCallHookPayload",
]

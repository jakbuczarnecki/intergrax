# © Artur Czarnecki. All rights reserved.

"""Typed Agent Governance REQUIRE_APPROVAL pause signal (UCA-6C-R6-R5.5)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.agent_runtime_governance import (
    PolicyEvaluationResult,
    ToolAuthorizationRequest,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)

SCHEMA_AGENT_GOVERNANCE_APPROVAL_PAUSE_SIGNAL_V1: Final = (
    "agent_governance_approval_pause_signal.v1"
)

_NON_EMPTY = Field(min_length=1)


class AgentGovernanceApprovalPauseSignal(BaseModel):
    """Authority-specific pause signal — not a permission grant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_governance_approval_pause_signal.v1"] = (
        SCHEMA_AGENT_GOVERNANCE_APPROVAL_PAUSE_SIGNAL_V1
    )
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    tenant_id: str = _NON_EMPTY
    agent_id: str = _NON_EMPTY
    tool_id: str = _NON_EMPTY
    step_id: str = _NON_EMPTY
    capability: str = _NON_EMPTY
    idempotency_key: str | None = None
    approval_id: str = _NON_EMPTY
    reason: str = _NON_EMPTY
    policy_results: tuple[PolicyEvaluationResult, ...] = ()
    policy_provenance_digest: str | None = None
    authorization_request: ToolAuthorizationRequest

    @field_validator(
        "tenant_id",
        "agent_id",
        "tool_id",
        "step_id",
        "capability",
        "approval_id",
        "reason",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("task_id", mode="before")
    @classmethod
    def _validate_task_id(cls, value: object) -> TaskId:
        return validate_task_id(value)

    @field_validator("run_id", mode="before")
    @classmethod
    def _validate_run_id(cls, value: object) -> RunId:
        return validate_run_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object) -> AttemptId:
        return validate_attempt_id(value)

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object) -> ExecutionId:
        return validate_execution_id(value)


__all__ = ["AgentGovernanceApprovalPauseSignal"]

# © Artur Czarnecki. All rights reserved.

"""Authority-specific Agent Runtime Governance human approval contracts (UCA-6C-R6-R5)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
from intergrax.contracts.lease_claim import LeaseOwnership
from intergrax.contracts.validation import validate_content_digest

SCHEMA_AGENT_GOVERNANCE_HUMAN_APPROVAL_REQUIREMENT_V1: Final = (
    "agent_governance_human_approval_requirement.v1"
)
SCHEMA_AGENT_GOVERNANCE_HUMAN_APPROVAL_PENDING_V1: Final = (
    "agent_governance_human_approval_pending.v1"
)
SCHEMA_AGENT_GOVERNANCE_HUMAN_APPROVAL_GRANT_V1: Final = (
    "agent_governance_human_approval_grant.v1"
)
SCHEMA_AGENT_GOVERNANCE_GRANT_LIFECYCLE_RECORD_V1: Final = (
    "agent_governance_grant_lifecycle_record.v1"
)

_NON_EMPTY = Field(min_length=1)


class AgentGovernanceGrantLifecycleState(StrEnum):
    """Frozen grant lifecycle partition (ADR-UCA-6C-ADR3-R2)."""

    AVAILABLE = "available"
    RESERVED = "reserved"
    APPLIED = "applied"
    TERMINAL = "terminal"


class LogicalInvocationFingerprint(BaseModel):
    """Stable digest binding one exact logical tool invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    digest: str = _NON_EMPTY

    @field_validator("digest")
    @classmethod
    def _validate_digest(cls, value: str) -> str:
        return validate_content_digest(value.strip())


class AgentGovernanceHumanApprovalRequirement(BaseModel):
    """Exact blocked invocation requirement at Agent Governance pause time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_governance_human_approval_requirement.v1"] = (
        SCHEMA_AGENT_GOVERNANCE_HUMAN_APPROVAL_REQUIREMENT_V1
    )
    agent_governance_invocation_scope_id: str = _NON_EMPTY
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    tenant_id: str = _NON_EMPTY
    agent_id: str = _NON_EMPTY
    tool_id: str = _NON_EMPTY
    step_id: str = _NON_EMPTY
    idempotency_key: str | None = None
    approval_id: str = _NON_EMPTY
    authorization_request: ToolAuthorizationRequest
    policy_results: tuple[PolicyEvaluationResult, ...] = ()
    policy_provenance_digest: str | None = None
    logical_invocation_fingerprint: LogicalInvocationFingerprint
    pause_generation: int = Field(ge=1)

    @field_validator(
        "agent_governance_invocation_scope_id",
        "tenant_id",
        "agent_id",
        "tool_id",
        "step_id",
        "approval_id",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("agent_governance_invocation_scope_id")
    @classmethod
    def _validate_agr_scope(cls, value: str) -> str:
        if not value.startswith("agr_"):
            raise ValueError("agent governance scope must use agr_ prefix")
        return value

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


class AgentGovernanceHumanApprovalPending(BaseModel):
    """Canonical pending Agent Governance approval on Task governance state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_governance_human_approval_pending.v1"] = (
        SCHEMA_AGENT_GOVERNANCE_HUMAN_APPROVAL_PENDING_V1
    )
    agent_governance_invocation_scope_id: str = _NON_EMPTY
    requirement: AgentGovernanceHumanApprovalRequirement
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    tenant_id: str = _NON_EMPTY
    agent_id: str = _NON_EMPTY
    tool_id: str = _NON_EMPTY
    step_id: str = _NON_EMPTY
    idempotency_key: str | None = None
    human_request_id: str = _NON_EMPTY
    pause_id: str = _NON_EMPTY
    policy_provenance_digest: str | None = None
    created_at: str = _NON_EMPTY
    generation: int = Field(ge=1)

    @field_validator(
        "agent_governance_invocation_scope_id",
        "tenant_id",
        "agent_id",
        "tool_id",
        "step_id",
        "human_request_id",
        "pause_id",
        "created_at",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @model_validator(mode="after")
    def _linkage_matches_requirement(self) -> AgentGovernanceHumanApprovalPending:
        req = self.requirement
        if (
            self.agent_governance_invocation_scope_id
            != req.agent_governance_invocation_scope_id
        ):
            raise ValueError("pending scope mismatch with requirement")
        if self.generation != req.pause_generation:
            raise ValueError("pending generation mismatch with requirement")
        return self


class AgentGovernanceHumanApprovalGrant(BaseModel):
    """Single-use Agent Governance approval artifact (not declarative/MSE grants)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_governance_human_approval_grant.v1"] = (
        SCHEMA_AGENT_GOVERNANCE_HUMAN_APPROVAL_GRANT_V1
    )
    grant_id: str = _NON_EMPTY
    agent_governance_invocation_scope_id: str = _NON_EMPTY
    pending_generation: int = Field(ge=1)
    logical_invocation_fingerprint: LogicalInvocationFingerprint
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    tenant_id: str = _NON_EMPTY
    agent_id: str = _NON_EMPTY
    tool_id: str = _NON_EMPTY
    step_id: str = _NON_EMPTY
    idempotency_key: str | None = None
    policy_provenance_digest: str | None = None
    human_request_id: str = _NON_EMPTY
    pause_id: str = _NON_EMPTY
    approved_at: str = _NON_EMPTY
    expires_at: str = _NON_EMPTY
    decided_by: str | None = None
    decision_metadata: tuple[str, ...] = ()

    @field_validator(
        "grant_id",
        "agent_governance_invocation_scope_id",
        "tenant_id",
        "agent_id",
        "tool_id",
        "step_id",
        "human_request_id",
        "pause_id",
        "approved_at",
        "expires_at",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class AgentGovernanceGrantReservation(BaseModel):
    """Durable RESERVED linkage for multi-host exclusivity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    logical_invocation_fingerprint: LogicalInvocationFingerprint
    pause_generation: int = Field(ge=1)
    agent_governance_invocation_scope_id: str = _NON_EMPTY
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    policy_provenance_digest: str | None = None
    ownership: LeaseOwnership

    @field_validator("agent_governance_invocation_scope_id")
    @classmethod
    def _strip_scope(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class AgentGovernanceGrantLifecycleRecord(BaseModel):
    """Grant artifact plus lifecycle state revision (canonical Task field payload)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_governance_grant_lifecycle_record.v1"] = (
        SCHEMA_AGENT_GOVERNANCE_GRANT_LIFECYCLE_RECORD_V1
    )
    grant: AgentGovernanceHumanApprovalGrant
    lifecycle_state: AgentGovernanceGrantLifecycleState
    lifecycle_revision: int = Field(ge=0)
    reservation: AgentGovernanceGrantReservation | None = None

    @model_validator(mode="after")
    def _reservation_state_consistent(self) -> AgentGovernanceGrantLifecycleRecord:
        if self.lifecycle_state is AgentGovernanceGrantLifecycleState.RESERVED:
            if self.reservation is None:
                raise ValueError("reserved grant requires reservation linkage")
        elif self.reservation is not None:
            raise ValueError("reservation only valid in reserved state")
        return self


def mint_agent_governance_invocation_scope_id() -> str:
    from uuid import uuid4

    return f"agr_{uuid4().hex}"


def digest_logical_invocation_fingerprint(
    *,
    task_id: str,
    run_id: str,
    attempt_id: str,
    execution_id: str,
    tenant_id: str,
    agent_id: str,
    tool_id: str,
    step_id: str,
    idempotency_key: str | None,
    payload_digest: str,
) -> LogicalInvocationFingerprint:
    import hashlib

    parts = (
        task_id,
        run_id,
        attempt_id,
        execution_id,
        tenant_id,
        agent_id,
        tool_id,
        step_id,
        idempotency_key or "",
        payload_digest,
    )
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return LogicalInvocationFingerprint(digest=f"sha256:{digest}")


__all__ = [
    "AgentGovernanceGrantLifecycleRecord",
    "AgentGovernanceGrantLifecycleState",
    "AgentGovernanceGrantReservation",
    "AgentGovernanceHumanApprovalGrant",
    "AgentGovernanceHumanApprovalPending",
    "AgentGovernanceHumanApprovalRequirement",
    "LogicalInvocationFingerprint",
    "digest_logical_invocation_fingerprint",
    "mint_agent_governance_invocation_scope_id",
]

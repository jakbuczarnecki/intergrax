# © Artur Czarnecki. All rights reserved.

"""Agent Runtime Governance contracts (NPSC-4).

Typed evaluation-only contracts for enterprise agent execution governance.
Governance decides *whether* an action is permitted; ExecutionRuntime decides *how* it runs.

Governance must NOT mint execution identity (RunId, AttemptId, ExecutionId).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

SCHEMA_AGENT_IDENTITY_V1: Final = "agent_identity.v1"
SCHEMA_CAPABILITY_GRANT_V1: Final = "capability_grant.v1"
SCHEMA_TOOL_AUTHORIZATION_REQUEST_V1: Final = "tool_authorization_request.v1"
SCHEMA_TOOL_AUTHORIZATION_DECISION_V1: Final = "tool_authorization_decision.v1"
SCHEMA_POLICY_EVALUATION_RESULT_V1: Final = "policy_evaluation_result.v1"
SCHEMA_APPROVAL_REQUEST_V1: Final = "approval_request.v1"
SCHEMA_GOVERNANCE_AUDIT_EVENT_V1: Final = "governance_audit_event.v1"

_NON_EMPTY = Field(min_length=1)


class ToolAuthorizationDecisionState(StrEnum):
    """Terminal or continuation governance decision for a tool invocation."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class ToolAuthorizationRiskLevel(StrEnum):
    """Conservative risk vocabulary for governance policy input."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalRequestStatus(StrEnum):
    """Human-in-the-loop approval lifecycle states."""

    CREATED = "created"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class AgentIdentity(BaseModel):
    """Logical agent identity for governance evaluation.

    Does NOT own or mint execution identity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_identity.v1"] = SCHEMA_AGENT_IDENTITY_V1
    agent_id: str = _NON_EMPTY
    tenant_id: str = _NON_EMPTY
    agent_version: str | None = None

    @field_validator("agent_id", "tenant_id")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("agent_version")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class CapabilityGrant(BaseModel):
    """Allowed capability scope for an agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_grant.v1"] = SCHEMA_CAPABILITY_GRANT_V1
    agent_id: str = _NON_EMPTY
    tenant_id: str = _NON_EMPTY
    allowed_capabilities: frozenset[str]
    denied_capabilities: frozenset[str] = frozenset()

    @field_validator("agent_id", "tenant_id")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @model_validator(mode="after")
    def _no_overlap(self) -> CapabilityGrant:
        overlap = self.allowed_capabilities & self.denied_capabilities
        if overlap:
            raise ValueError(
                f"capabilities cannot be both allowed and denied: {sorted(overlap)}"
            )
        return self

    def is_capability_allowed(self, capability: str) -> bool:
        normalized = capability.strip()
        if not normalized:
            return False
        if normalized in self.denied_capabilities:
            return False
        return normalized in self.allowed_capabilities


class ToolAuthorizationRequest(BaseModel):
    """Explicit governance request: agent wants to execute a tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tool_authorization_request.v1"] = (
        SCHEMA_TOOL_AUTHORIZATION_REQUEST_V1
    )
    agent: AgentIdentity
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId | None = None
    capability: str = _NON_EMPTY
    tool_id: str = _NON_EMPTY
    requested_action: str = _NON_EMPTY
    risk_level: ToolAuthorizationRiskLevel
    approval_evidence_ref: str | None = None

    @field_validator("capability", "tool_id", "requested_action")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("approval_evidence_ref")
    @classmethod
    def _strip_optional_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

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
    def _validate_execution_id(cls, value: object) -> ExecutionId | None:
        if value is None:
            return None
        return validate_execution_id(value)


class PolicyEvaluationResult(BaseModel):
    """Explainable result from a single policy provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["policy_evaluation_result.v1"] = (
        SCHEMA_POLICY_EVALUATION_RESULT_V1
    )
    policy_id: str = _NON_EMPTY
    decision: ToolAuthorizationDecisionState
    reason: str = _NON_EMPTY

    @field_validator("policy_id", "reason")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class ToolAuthorizationDecision(BaseModel):
    """Governance decision for a tool authorization request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tool_authorization_decision.v1"] = (
        SCHEMA_TOOL_AUTHORIZATION_DECISION_V1
    )
    decision: ToolAuthorizationDecisionState
    reason: str = _NON_EMPTY
    policy_results: tuple[PolicyEvaluationResult, ...] = ()

    @field_validator("reason")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @property
    def is_terminal_deny(self) -> bool:
        return self.decision is ToolAuthorizationDecisionState.DENY

    @property
    def requires_approval(self) -> bool:
        return self.decision is ToolAuthorizationDecisionState.REQUIRE_APPROVAL

    @property
    def is_allowed(self) -> bool:
        return self.decision is ToolAuthorizationDecisionState.ALLOW


class ApprovalRequest(BaseModel):
    """Human-in-the-loop decision contract for governed tool execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["approval_request.v1"] = SCHEMA_APPROVAL_REQUEST_V1
    approval_id: str = _NON_EMPTY
    authorization_request: ToolAuthorizationRequest
    status: ApprovalRequestStatus
    risk_classification: ToolAuthorizationRiskLevel
    requested_at: datetime
    expires_at: datetime
    decided_at: datetime | None = None
    decided_by: str | None = None
    audit_metadata: tuple[str, ...] = ()

    @field_validator("approval_id", "decided_by")
    @classmethod
    def _strip_optional_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def _decision_fields_consistent(self) -> ApprovalRequest:
        if self.status in (
            ApprovalRequestStatus.APPROVED,
            ApprovalRequestStatus.REJECTED,
        ):
            if self.decided_at is None or self.decided_by is None:
                raise ValueError(
                    "approved/rejected status requires decided_at and decided_by"
                )
        return self


class GovernanceAuditEvent(BaseModel):
    """Immutable audit record for every governance decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["governance_audit_event.v1"] = (
        SCHEMA_GOVERNANCE_AUDIT_EVENT_V1
    )
    event_id: str = _NON_EMPTY
    execution_id: ExecutionId | None = None
    run_id: RunId
    attempt_id: AttemptId
    task_id: TaskId
    agent_id: str = _NON_EMPTY
    capability: str = _NON_EMPTY
    tool_id: str = _NON_EMPTY
    decision: ToolAuthorizationDecisionState
    policy_results: tuple[PolicyEvaluationResult, ...] = ()
    timestamp: datetime

    @field_validator("event_id", "agent_id", "capability", "tool_id")
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
    def _validate_execution_id(cls, value: object) -> ExecutionId | None:
        if value is None:
            return None
        return validate_execution_id(value)


def mint_governance_audit_event_id() -> str:
    """Mint a governance audit event identifier (NOT execution identity)."""
    return f"governance_evt_{uuid4().hex}"


class AgentRuntimePolicyProvider(Protocol):
    """Plugin policy provider evaluated by the governance policy engine."""

    @property
    def policy_id(self) -> str: ...

    def evaluate(
        self,
        request: ToolAuthorizationRequest,
    ) -> PolicyEvaluationResult: ...


class CapabilityGrantResolverPort(Protocol):
    """Resolves capability grants for an agent identity."""

    def resolve_grant(self, agent: AgentIdentity) -> CapabilityGrant | None: ...


class ApprovalStorePort(Protocol):
    """Persistent approval state for HITL continuation."""

    def create(self, approval: ApprovalRequest) -> ApprovalRequest: ...

    def get(self, approval_id: str) -> ApprovalRequest | None: ...

    def update(self, approval: ApprovalRequest) -> ApprovalRequest: ...


class GovernanceAuditSinkPort(Protocol):
    """Records governance audit events."""

    def record(self, event: GovernanceAuditEvent) -> None: ...

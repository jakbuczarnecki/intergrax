# © Artur Czarnecki. All rights reserved.

"""Multi-agent coordination governance contracts (NPSC-5D/R1).

Semantic coordination admission evaluated through canonical ``PolicyDecision`` /
``PolicyAction`` — not a second policy engine or NPSC-local authorization runtime.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.control_plane_mutation import GovernanceEvaluationPoint
from intergrax.contracts.evaluated_policy_decision import request_digest_for_payload
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

SCHEMA_MULTI_AGENT_COORDINATION_GOVERNANCE_REQUEST_V1: Final = (
    "multi_agent_coordination_governance_request.v1"
)
SCHEMA_MULTI_AGENT_COORDINATION_GOVERNANCE_EVIDENCE_V1: Final = (
    "multi_agent_coordination_governance_evidence.v1"
)

_NON_EMPTY = Field(min_length=1)


class MultiAgentCoordinationExecutionMode(StrEnum):
    """Semantic coordination shape for governance — mirrors NPSC-5C modes."""

    SINGLE = "single"
    FAN_OUT = "fan_out"


class MultiAgentCoordinationCapabilityKind(StrEnum):
    """Capability authority shape without Agent Distribution implementation types."""

    UNRESOLVED_TASK = "unresolved_task"
    RESOLVED_REQUIREMENT = "resolved_requirement"


class MultiAgentCoordinationGovernanceContribution(BaseModel):
    """One semantic contribution fact for coordination-level admission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contribution_id: str = _NON_EMPTY
    capability_kind: MultiAgentCoordinationCapabilityKind
    required_capability_ids: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("contribution_id")
    @classmethod
    def _strip_contribution_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("required_capability_ids")
    @classmethod
    def _normalize_capability_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized = tuple(
            sorted({item.strip() for item in value if item.strip()}),
        )
        return normalized


class MultiAgentCoordinationGovernanceRequest(BaseModel):
    """Stage-1 semantic coordination admission request — no physical agent identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["multi_agent_coordination_governance_request.v1"] = (
        SCHEMA_MULTI_AGENT_COORDINATION_GOVERNANCE_REQUEST_V1
    )
    evaluation_point: GovernanceEvaluationPoint = (
        GovernanceEvaluationPoint.MULTI_AGENT_COORDINATION
    )
    intent_id: str = _NON_EMPTY
    execution_mode: MultiAgentCoordinationExecutionMode
    contributions: tuple[MultiAgentCoordinationGovernanceContribution, ...] = Field(
        min_length=1,
    )
    requested_max_concurrency: int | None = None
    task_scope_id: str = _NON_EMPTY
    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    principal: RequestIdentity

    @field_validator(
        "intent_id",
        "task_scope_id",
        "application_id",
        "application_environment_id",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("requested_max_concurrency")
    @classmethod
    def _validate_requested_max_concurrency(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("requested_max_concurrency must be positive when set")
        return value

    @property
    def tenant_id(self) -> str:
        return self.principal.tenant_id


class MultiAgentCoordinationGovernanceEvidence(BaseModel):
    """Typed authorization provenance for coordination admission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["multi_agent_coordination_governance_evidence.v1"] = (
        SCHEMA_MULTI_AGENT_COORDINATION_GOVERNANCE_EVIDENCE_V1
    )
    evaluation_point: GovernanceEvaluationPoint = (
        GovernanceEvaluationPoint.MULTI_AGENT_COORDINATION
    )
    request_digest: str = _NON_EMPTY
    intent_id: str = _NON_EMPTY
    execution_mode: MultiAgentCoordinationExecutionMode
    contribution_count: int = Field(ge=1)
    tenant_id: str = _NON_EMPTY
    task_scope_id: str = _NON_EMPTY
    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    policy_action: PolicyAction
    policy_rule_id: str = ""
    policy_decision_id: str = ""

    @field_validator("request_digest")
    @classmethod
    def _validate_request_digest(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized.startswith("sha256:"):
            raise ValueError("request_digest_must_be_sha256")
        return normalized


class MultiAgentCoordinationGovernanceResult(BaseModel):
    """Evaluation-only coordination admission outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    permitted: bool
    decision: PolicyDecision
    evidence: MultiAgentCoordinationGovernanceEvidence
    requires_governed_continuation: bool = False
    validation_failed: bool = False


class MultiAgentCoordinationGovernancePolicyRule:
    """Immutable runtime rule for coordination admission evaluation."""

    __slots__ = ("rule_id", "decision", "execution_mode", "reason")

    def __init__(
        self,
        *,
        rule_id: str,
        decision: PolicyAction,
        execution_mode: MultiAgentCoordinationExecutionMode | None = None,
        reason: str = "",
    ) -> None:
        normalized_rule_id = rule_id.strip()
        if not normalized_rule_id:
            raise ValueError("rule_id must be non-empty")
        if not isinstance(decision, PolicyAction):
            raise TypeError("decision must be PolicyAction")
        self.rule_id = normalized_rule_id
        self.decision = decision
        self.execution_mode = execution_mode
        self.reason = reason.strip()


class MultiAgentCoordinationGovernanceEvaluator(Protocol):
    """Configured policy evaluator for semantic coordination admission."""

    def evaluate(
        self,
        request: MultiAgentCoordinationGovernanceRequest,
    ) -> PolicyDecision:
        """Return a fresh governance decision for ``request``."""
        ...


class MultiAgentCoordinationGovernancePort(Protocol):
    """Public governance admission boundary for multi-agent coordination callers."""

    def evaluate(
        self,
        request: MultiAgentCoordinationGovernanceRequest,
    ) -> MultiAgentCoordinationGovernanceResult:
        """Evaluate coordination admission using typed request facts only."""
        ...


def multi_agent_coordination_governance_request_digest(
    request: MultiAgentCoordinationGovernanceRequest,
) -> str:
    payload = request.model_dump(mode="json")
    return request_digest_for_payload(payload)


def evidence_from_request_and_decision(
    request: MultiAgentCoordinationGovernanceRequest,
    *,
    decision: PolicyDecision,
    request_digest: str,
) -> MultiAgentCoordinationGovernanceEvidence:
    return MultiAgentCoordinationGovernanceEvidence(
        request_digest=request_digest,
        intent_id=request.intent_id,
        execution_mode=request.execution_mode,
        contribution_count=len(request.contributions),
        tenant_id=request.tenant_id,
        task_scope_id=request.task_scope_id,
        application_id=request.application_id,
        application_environment_id=request.application_environment_id,
        policy_action=decision.action,
        policy_rule_id=decision.policy_rule_id,
        policy_decision_id=decision.decision_id,
    )

# © Artur Czarnecki. All rights reserved.

"""Canonical control-plane mutation authorization contract (CLA-04 foundation).

Describes a proposed state-changing control-plane operation for policy evaluation.
Domain owners execute mutations; this contract is evaluation-only input.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.evaluated_policy_decision import request_digest_for_payload
from intergrax.contracts.execution_identity import (
    TaskId,
    RunId,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

SCHEMA_CONTROL_PLANE_MUTATION_REQUEST_V1: Final = "control_plane_mutation_request.v1"
SCHEMA_CONTROL_PLANE_MUTATION_AUTHORIZATION_EVIDENCE_V1: Final = (
    "control_plane_mutation_authorization_evidence.v1"
)

_NON_EMPTY = Field(min_length=1)


class GovernanceEvaluationPoint(StrEnum):
    """Minimal Governed Execution evaluation-point taxonomy."""

    CONTROL_PLANE_MUTATION = "control_plane_mutation"


class ControlPlaneMutationRisk(StrEnum):
    """Cross-domain conservative risk vocabulary for policy input."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ControlPlaneMutationRequest(BaseModel):
    """Provider-neutral proposed control-plane mutation for authorization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["control_plane_mutation_request.v1"] = (
        SCHEMA_CONTROL_PLANE_MUTATION_REQUEST_V1
    )
    mutation_id: str = _NON_EMPTY
    mutation_type: str = _NON_EMPTY
    principal: RequestIdentity
    resource_scope: str = _NON_EMPTY
    resource_type: str = _NON_EMPTY
    resource_id: str = _NON_EMPTY
    current_revision: str = _NON_EMPTY
    target_revision: str = _NON_EMPTY
    risk_classification: ControlPlaneMutationRisk
    approval_evidence_ref: str | None = None
    task_id: TaskId | None = None
    run_id: RunId | None = None

    @field_validator(
        "mutation_id",
        "mutation_type",
        "resource_scope",
        "resource_type",
        "resource_id",
        "current_revision",
        "target_revision",
    )
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
    def _validate_task_id(cls, value: object) -> TaskId | None:
        if value is None:
            return None
        return validate_task_id(value)

    @field_validator("run_id", mode="before")
    @classmethod
    def _validate_run_id(cls, value: object) -> RunId | None:
        if value is None:
            return None
        return validate_run_id(value)

    @model_validator(mode="after")
    def _execution_identity_pair(self) -> ControlPlaneMutationRequest:
        if (self.task_id is None) ^ (self.run_id is None):
            raise ValueError("task_id and run_id must both be set or both absent")
        return self

    @property
    def tenant_id(self) -> str:
        return self.principal.tenant_id


class ControlPlaneMutationAuthorizationScope(BaseModel):
    """Exact scope identity for canonical HITL / governed continuation wiring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mutation_id: str = _NON_EMPTY
    mutation_type: str = _NON_EMPTY
    tenant_id: str = _NON_EMPTY
    resource_scope: str = _NON_EMPTY
    resource_type: str = _NON_EMPTY
    resource_id: str = _NON_EMPTY
    current_revision: str = _NON_EMPTY
    target_revision: str = _NON_EMPTY
    task_id: str | None = None
    run_id: str | None = None

    @field_validator(
        "mutation_id",
        "mutation_type",
        "tenant_id",
        "resource_scope",
        "resource_type",
        "resource_id",
        "current_revision",
        "target_revision",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class ControlPlaneMutationAuthorizationEvidence(BaseModel):
    """Typed authorization provenance for later proof projection (R5)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["control_plane_mutation_authorization_evidence.v1"] = (
        SCHEMA_CONTROL_PLANE_MUTATION_AUTHORIZATION_EVIDENCE_V1
    )
    evaluation_point: GovernanceEvaluationPoint = (
        GovernanceEvaluationPoint.CONTROL_PLANE_MUTATION
    )
    request_digest: str = _NON_EMPTY
    mutation_id: str = _NON_EMPTY
    mutation_type: str = _NON_EMPTY
    tenant_id: str = _NON_EMPTY
    resource_scope: str = _NON_EMPTY
    resource_type: str = _NON_EMPTY
    resource_id: str = _NON_EMPTY
    current_revision: str = _NON_EMPTY
    target_revision: str = _NON_EMPTY
    risk_classification: ControlPlaneMutationRisk
    principal_type: PrincipalType
    principal_user_id: str | None = None
    principal_auth_subject: str | None = None
    task_id: str | None = None
    run_id: str | None = None
    approval_evidence_ref: str | None = None
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


class ControlPlaneMutationAuthorizationResult(BaseModel):
    """Evaluation-only authorization outcome — does not execute domain mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    permitted: bool
    decision: PolicyDecision
    evidence: ControlPlaneMutationAuthorizationEvidence
    requires_governed_continuation: bool = False
    authorization_scope: ControlPlaneMutationAuthorizationScope | None = None
    validation_failed: bool = False


class ControlPlaneMutationPolicyEvaluator(Protocol):
    """Configured policy/authority evaluator for control-plane mutations."""

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        """Return a fresh governance decision for ``request``."""


def control_plane_mutation_request_digest(
    request: ControlPlaneMutationRequest,
) -> str:
    """Canonical digest binding principal, resource, revisions, and mutation identity."""
    payload = request.model_dump(mode="json")
    return request_digest_for_payload(payload)


def authorization_scope_for_request(
    request: ControlPlaneMutationRequest,
) -> ControlPlaneMutationAuthorizationScope:
    return ControlPlaneMutationAuthorizationScope(
        mutation_id=request.mutation_id,
        mutation_type=request.mutation_type,
        tenant_id=request.tenant_id,
        resource_scope=request.resource_scope,
        resource_type=request.resource_type,
        resource_id=request.resource_id,
        current_revision=request.current_revision,
        target_revision=request.target_revision,
        task_id=str(request.task_id) if request.task_id is not None else None,
        run_id=str(request.run_id) if request.run_id is not None else None,
    )


def evidence_from_request_and_decision(
    request: ControlPlaneMutationRequest,
    *,
    decision: PolicyDecision,
    request_digest: str,
) -> ControlPlaneMutationAuthorizationEvidence:
    return ControlPlaneMutationAuthorizationEvidence(
        request_digest=request_digest,
        mutation_id=request.mutation_id,
        mutation_type=request.mutation_type,
        tenant_id=request.tenant_id,
        resource_scope=request.resource_scope,
        resource_type=request.resource_type,
        resource_id=request.resource_id,
        current_revision=request.current_revision,
        target_revision=request.target_revision,
        risk_classification=request.risk_classification,
        principal_type=request.principal.principal_type,
        principal_user_id=request.principal.user_id,
        principal_auth_subject=request.principal.auth_subject,
        task_id=str(request.task_id) if request.task_id is not None else None,
        run_id=str(request.run_id) if request.run_id is not None else None,
        approval_evidence_ref=request.approval_evidence_ref,
        policy_action=decision.action,
        policy_rule_id=decision.policy_rule_id,
        policy_decision_id=decision.decision_id,
    )

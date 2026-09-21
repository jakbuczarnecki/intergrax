# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed Governance decision facts for the Evidence Plane (GR-8).

Governance owns permission semantics; this contract records immutable facts only.
Evidence persistence must never return or alter ``PolicyDecision``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.canonical_payload_hash import stable_payload_hash
from intergrax.contracts.decision_governance_material import DecisionGovernanceMaterialRef
from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    ExecutionId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

SCHEMA_GOVERNED_EXECUTION_GOVERNANCE_DECISION_FACT_V1: Final = (
    "governed_execution_governance_decision_fact.v1"
)
_NON_EMPTY = Field(min_length=1)


class GovernedExecutionEvaluationPoint(StrEnum):
    """Canonical Governed Execution evaluation points (GOVERNED_EXECUTION §G3B)."""

    ROOT_EXECUTION_ADMISSION = "root_execution_admission"
    AGENT_DECISION = "agent_decision"
    INTERRUPT = "interrupt"
    PRE_MODEL = "pre_model"
    TOOL_PLAN_OR_ACCESS = "tool_plan_or_access"
    TOOL_INVOCATION_AUTHORIZATION = "tool_invocation_authorization"
    TOOL_INVOCATION_POLICY = "tool_invocation_policy"
    MEANINGFUL_SIDE_EFFECT = "meaningful_side_effect"
    PRE_OUTPUT = "pre_output"
    POST_RUN = "post_run"
    CONTROL_PLANE_MUTATION = "control_plane_mutation"


class GovernanceDecisionEvidenceFact(BaseModel):
    """Immutable Governance decision fact — append-only evidence projection input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["governed_execution_governance_decision_fact.v1"] = (
        SCHEMA_GOVERNED_EXECUTION_GOVERNANCE_DECISION_FACT_V1
    )
    evidence_id: str = _NON_EMPTY
    recorded_at: datetime
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    principal_id: str = _NON_EMPTY
    task_id: TaskId | None = None
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None
    execution_id: ExecutionId | None = None
    evaluation_point: GovernedExecutionEvaluationPoint
    action: str = _NON_EMPTY
    resource_type: str = ""
    resource_scope: str = ""
    decision: PolicyAction
    reason: str = ""
    reason_code: str = ""
    policy_bundle_id: str = ""
    policy_bundle_version: str = ""
    policy_bundle_digest: str = ""
    policy_rule_id: str = ""
    request_digest: str = _NON_EMPTY
    idempotency_key: str = _NON_EMPTY
    decision_material_ref: DecisionGovernanceMaterialRef | None = None
    human_review_evidence_ref: str | None = None

    @field_validator(
        "evidence_id",
        "tenant_id",
        "workspace_id",
        "principal_id",
        "action",
        "request_digest",
        "idempotency_key",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

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

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object) -> AttemptId | None:
        if value is None:
            return None
        return validate_attempt_id(value)

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object) -> ExecutionId | None:
        if value is None:
            return None
        return validate_execution_id(value)

    @field_validator("request_digest")
    @classmethod
    def _validate_request_digest(cls, value: str) -> str:
        if not value.startswith("sha256:"):
            raise ValueError("request_digest_must_be_sha256")
        return value

    @model_validator(mode="after")
    def _validate_bundle_provenance(self) -> GovernanceDecisionEvidenceFact:
        has_any = any(
            (
                self.policy_bundle_id,
                self.policy_bundle_version,
                self.policy_bundle_digest,
            )
        )
        if has_any and not (
            self.policy_bundle_id and self.policy_bundle_version and self.policy_bundle_digest
        ):
            raise ValueError("policy_bundle_provenance_incomplete")
        if self.policy_bundle_digest and not self.policy_bundle_digest.startswith("sha256:"):
            raise ValueError("policy_bundle_digest_must_be_sha256")
        return self

    @property
    def has_full_execution_correlation(self) -> bool:
        return (
            self.task_id is not None
            and self.run_id is not None
            and self.attempt_id is not None
            and self.execution_id is not None
        )


class GovernanceEvidencePersistenceOutcome(BaseModel):
    """Non-authoritative persistence acknowledgment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    persisted: bool
    evidence_id: str = _NON_EMPTY
    error_code: str | None = None


@runtime_checkable
class GovernanceEvidencePersistencePort(Protocol):
    """Append-only Governance fact persistence — never returns policy decisions."""

    def persist(self, fact: GovernanceDecisionEvidenceFact) -> GovernanceEvidencePersistenceOutcome:
        """Persist one immutable fact. Idempotent on ``evidence_id`` / idempotency key."""
        ...


def governance_evidence_id_from_idempotency(idempotency_key: str) -> str:
    digest = stable_payload_hash({"idempotency_key": idempotency_key})
    suffix = digest.removeprefix("sha256:")[:32]
    return f"gov_ev_{suffix}"


def deterministic_runtime_event_id_for_governance_fact(fact: GovernanceDecisionEvidenceFact) -> EventId:
    digest = stable_payload_hash(
        {
            "schema": fact.schema_version,
            "evidence_id": fact.evidence_id,
            "idempotency_key": fact.idempotency_key,
        }
    )
    suffix = digest.removeprefix("sha256:")[:32]
    return EventId(f"evt_{suffix}")


def build_governance_fact_from_policy_decision(
    *,
    evaluation_point: GovernedExecutionEvaluationPoint,
    tenant_id: str,
    workspace_id: str,
    principal_id: str,
    decision: PolicyDecision,
    request_digest: str,
    idempotency_key: str,
    action: str,
    resource_type: str = "",
    resource_scope: str = "",
    task_id: TaskId | None = None,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
    execution_id: ExecutionId | None = None,
    decision_material_ref: DecisionGovernanceMaterialRef | None = None,
    human_review_evidence_ref: str | None = None,
    recorded_at: datetime | None = None,
) -> GovernanceDecisionEvidenceFact:
    """Project a canonical ``PolicyDecision`` into a typed Governance evidence fact."""
    if decision.action not in (
        PolicyAction.ALLOW,
        PolicyAction.DENY,
        PolicyAction.REQUIRE_HUMAN,
        PolicyAction.ESCALATE,
    ):
        raise ValueError(
            "governance_evidence_requires_allow_deny_require_human_or_escalate",
        )
    evidence_id = governance_evidence_id_from_idempotency(idempotency_key)
    return GovernanceDecisionEvidenceFact(
        evidence_id=evidence_id,
        recorded_at=recorded_at or datetime.now(timezone.utc),
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        principal_id=principal_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        evaluation_point=evaluation_point,
        action=action,
        resource_type=resource_type,
        resource_scope=resource_scope,
        decision=decision.action,
        reason=decision.reason,
        reason_code="",
        policy_bundle_id=decision.policy_bundle_id or "",
        policy_bundle_version=decision.policy_bundle_version or "",
        policy_bundle_digest=decision.policy_bundle_digest or "",
        policy_rule_id=decision.policy_rule_id or "",
        request_digest=request_digest,
        idempotency_key=idempotency_key,
        decision_material_ref=decision_material_ref,
        human_review_evidence_ref=human_review_evidence_ref,
    )


__all__ = [
    "GovernanceDecisionEvidenceFact",
    "GovernanceEvidencePersistenceOutcome",
    "GovernanceEvidencePersistencePort",
    "GovernedExecutionEvaluationPoint",
    "SCHEMA_GOVERNED_EXECUTION_GOVERNANCE_DECISION_FACT_V1",
    "build_governance_fact_from_policy_decision",
    "deterministic_runtime_event_id_for_governance_fact",
    "governance_evidence_id_from_idempotency",
]

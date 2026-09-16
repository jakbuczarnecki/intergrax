# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Immutable Decision material bound to canonical Governance authorization (GR-6).

Typed provenance for decision-derived consequential effects. Decision assessment
only — final side-effect permission remains ``PolicyAction`` via
``MeaningfulSideEffectAuthorizationBoundary``.
"""

from __future__ import annotations

from typing import Final, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.decision_authorization import (
    DecisionExecutionAction,
    DecisionExecutionAuthorization,
    decision_execution_action,
    validate_decision_execution_action_kind,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionId,
    DecisionVersion,
    validate_decision_id,
    validate_decision_tenant_id,
    validate_decision_version,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    DecisionBranchId,
    validate_decision_branch_id,
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
from intergrax.contracts.validation import compute_sha256_content_digest, validate_content_digest

SCHEMA_DECISION_GOVERNANCE_MATERIAL_REF_V1: Final = "decision_governance_material_ref.v1"

_NON_EMPTY = Field(min_length=1)


class DecisionGovernanceMaterialMismatchError(ValueError):
    """Fail-closed binding violation between Decision material and Governance scope."""


class DecisionGovernanceMaterialRef(BaseModel):
    """Exact immutable decision snapshot referenced by one governance authorization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["decision_governance_material_ref.v1"] = (
        SCHEMA_DECISION_GOVERNANCE_MATERIAL_REF_V1
    )
    tenant_id: str = _NON_EMPTY
    decision_id: DecisionId
    decision_version: int = Field(ge=1)
    decision_branch_id: DecisionBranchId
    decision_material_digest: str = _NON_EMPTY
    bound_action_kind: str = _NON_EMPTY
    bound_action_subject: str = _NON_EMPTY
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId

    @field_validator("tenant_id", "bound_action_subject", "decision_material_digest")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("decision_id", mode="before")
    @classmethod
    def _validate_decision_id(cls, value: object) -> DecisionId:
        return validate_decision_id(value)

    @field_validator("decision_version", mode="before")
    @classmethod
    def _validate_decision_version(cls, value: object) -> int:
        return validate_decision_version(value)

    @field_validator("decision_branch_id", mode="before")
    @classmethod
    def _validate_branch(cls, value: object) -> DecisionBranchId:
        return validate_decision_branch_id(value)

    @field_validator("bound_action_kind", mode="before")
    @classmethod
    def _validate_action_kind(cls, value: object) -> str:
        validated = validate_decision_execution_action_kind(value)
        return str(validated)

    @field_validator("decision_material_digest")
    @classmethod
    def _validate_digest(cls, value: str) -> str:
        return validate_content_digest(value.strip())

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

    @model_validator(mode="after")
    def _validate_tenant(self) -> DecisionGovernanceMaterialRef:
        validate_decision_tenant_id(self.tenant_id)
        return self

    @property
    def decision_version_value(self) -> DecisionVersion:
        return DecisionVersion(self.decision_version)

    @property
    def bound_action(self) -> DecisionExecutionAction:
        return decision_execution_action(
            kind=self.bound_action_kind,
            subject=self.bound_action_subject,
        )

    @property
    def execution_lineage(self) -> DecisionExecutionLineage:
        return DecisionExecutionLineage(
            task_id=self.task_id,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
            execution_id=self.execution_id,
        )


T = TypeVar("T")


def compute_decision_governance_material_digest(
    *,
    decision: AuthoritativeAcceptedDecision[T],
    action: DecisionExecutionAction,
) -> str:
    """Stable digest over identity, lineage, artifact kind, and bound action."""
    if type(decision) is not AuthoritativeAcceptedDecision:
        raise TypeError("decision must be AuthoritativeAcceptedDecision")
    if type(action) is not DecisionExecutionAction:
        raise TypeError("action must be DecisionExecutionAction")
    execution = decision.identity.execution
    if execution.execution_id is None:
        raise ValueError(
            "decision execution lineage must include execution_id for governance material digest",
        )
    canonical = "|".join(
        (
            decision.identity.tenant_id,
            str(decision.identity.decision_id),
            str(decision.identity.version.value),
            str(decision.lineage.current.branch_id),
            str(execution.task_id),
            str(execution.run_id),
            str(execution.attempt_id),
            str(execution.execution_id),
            str(action.kind),
            action.subject,
            str(decision.artifact.kind),
        ),
    ).encode("utf-8")
    return compute_sha256_content_digest(canonical)


def decision_governance_material_ref_from_accepted(
    *,
    decision: AuthoritativeAcceptedDecision[T],
    action: DecisionExecutionAction,
) -> DecisionGovernanceMaterialRef:
    """Build immutable material ref from one authoritative accepted decision."""
    execution = decision.identity.execution
    if execution.execution_id is None:
        raise ValueError(
            "decision execution lineage must include execution_id for governance material",
        )
    digest = compute_decision_governance_material_digest(
        decision=decision,
        action=action,
    )
    return DecisionGovernanceMaterialRef(
        tenant_id=decision.identity.tenant_id,
        decision_id=decision.identity.decision_id,
        decision_version=decision.identity.version.value,
        decision_branch_id=decision.lineage.current.branch_id,
        decision_material_digest=digest,
        bound_action_kind=str(action.kind),
        bound_action_subject=action.subject,
        task_id=execution.task_id,
        run_id=execution.run_id,
        attempt_id=execution.attempt_id,
        execution_id=execution.execution_id,
    )


def validate_decision_governance_material_for_decision(
    *,
    material: DecisionGovernanceMaterialRef,
    decision: AuthoritativeAcceptedDecision[T],
    action: DecisionExecutionAction,
) -> None:
    """Reject stale or recomputed decision relative to bound material."""
    expected = decision_governance_material_ref_from_accepted(
        decision=decision,
        action=action,
    )
    if material != expected:
        raise DecisionGovernanceMaterialMismatchError(
            "decision governance material does not match authoritative decision and action",
        )


def validate_decision_governance_material_for_authorization(
    *,
    material: DecisionGovernanceMaterialRef,
    authorization: DecisionExecutionAuthorization,
    action: DecisionExecutionAction,
) -> None:
    """Reject authorization reuse when decision material diverges."""
    if authorization.tenant_id != material.tenant_id:
        raise DecisionGovernanceMaterialMismatchError(
            "decision governance material tenant_id must match execution authorization",
        )
    ref = authorization.decision_ref
    if str(ref.identity.decision_id) != str(material.decision_id):
        raise DecisionGovernanceMaterialMismatchError(
            "decision governance material decision_id must match authorization",
        )
    if ref.identity.version.value != material.decision_version:
        raise DecisionGovernanceMaterialMismatchError(
            "decision governance material decision_version must match authorization",
        )
    if ref.lineage_ref.branch_id != material.decision_branch_id:
        raise DecisionGovernanceMaterialMismatchError(
            "decision governance material branch_id must match authorization",
        )
    if authorization.action != action or authorization.action != material.bound_action:
        raise DecisionGovernanceMaterialMismatchError(
            "decision governance material action must match execution authorization",
        )


__all__ = [
    "DecisionGovernanceMaterialMismatchError",
    "DecisionGovernanceMaterialRef",
    "SCHEMA_DECISION_GOVERNANCE_MATERIAL_REF_V1",
    "compute_decision_governance_material_digest",
    "decision_governance_material_ref_from_accepted",
    "validate_decision_governance_material_for_authorization",
    "validate_decision_governance_material_for_decision",
]

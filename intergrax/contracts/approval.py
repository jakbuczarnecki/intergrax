# © Artur Czarnecki. All rights reserved.

"""Multiplayer Approval domain contracts (MP-4C).

Neutral typed contracts for Approval identity, request lifecycle, human
approval actions, and approval outcomes. Distinct from:

- ``Decision`` / ``DecisionOutcome`` — Decision Domain (referenced only).
- ``WorkItem`` / ``WorkArtifact`` / ``WorkArtifactVersion`` — Collaborative Work.
- ``Task`` / ``Run`` / ``Attempt`` / ``Execution`` — Unified Execution Runtime.
- Governed Execution HITL pause/resume — bridge consumer only; not owned here.

No persistence, services, workflow engine, or runtime orchestration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final, Literal, NewType, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.collaborative_work import (
    AuthorityDelegation,
    MembershipResolutionMode,
    WorkArtifactVersionRef,
    WorkspaceMembership,
)
from intergrax.contracts.decision import DecisionId, validate_decision_id
from intergrax.contracts.execution_identity import (
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.execution_provenance import ExecutionProvenanceRef

SCHEMA_APPROVAL_REQUEST_V1: Final = "approval_request.v1"
SCHEMA_APPROVAL_REFERENCES_V1: Final = "approval_references.v1"
SCHEMA_HUMAN_APPROVAL_ACTION_V1: Final = "human_approval_action.v1"
SCHEMA_APPROVAL_OUTCOME_V1: Final = "approval_outcome.v1"
SCHEMA_CREATE_APPROVAL_REQUEST_V1: Final = "create_approval_request.v1"
SCHEMA_EXECUTE_HUMAN_APPROVAL_ACTION_REQUEST_V1: Final = (
    "execute_human_approval_action_request.v1"
)

ApprovalId = NewType("ApprovalId", str)

_APPROVAL_ID_PREFIX = "approval_"
_DECISION_ID_PREFIX = "decision_"
_CANONICAL_SUFFIX = re.compile(r"^[0-9a-f]{32}$")
_NON_EMPTY = Field(min_length=1)

_EXECUTION_ID_PREFIXES: Final = frozenset(
    {
        "task_",
        "run_",
        "attempt_",
        "exec_",
        "evt_",
    },
)


class ApprovalContractInvariantError(Exception):
    """Base invariant violation for MP-4C Approval contracts."""


class ApprovalScopeInvariantError(ApprovalContractInvariantError):
    """Tenant/workspace scope or cross-reference scope mismatch."""


class ApprovalIdentityInvariantError(ApprovalContractInvariantError):
    """Approval identity field validation failure."""


class ApprovalLifecycleTransitionError(ApprovalContractInvariantError):
    """Unsupported or invalid Approval lifecycle transition."""


class ApprovalReferenceInvariantError(ApprovalContractInvariantError):
    """Invalid or inconsistent approval outbound reference."""


class ApprovalLifecycleState(StrEnum):
    """Explicit Approval request lifecycle — not Decision or Execution state."""

    REQUESTED = "requested"
    ASSIGNED = "assigned"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class HumanApprovalActionType(StrEnum):
    """Human action verbs — distinct from ``ApprovalLifecycleState`` values."""

    ASSIGN = "assign"
    START_REVIEW = "start_review"
    SUBMIT_COMMENT = "submit_comment"
    CONFIRM_APPROVAL = "confirm_approval"
    CONFIRM_REJECTION = "confirm_rejection"
    CANCEL = "cancel"
    EXPIRE = "expire"


class ApprovalOutcomeDisposition(StrEnum):
    """Substantive approval outcome — distinct from lifecycle terminal states."""

    APPROVED = "approved"
    REJECTED = "rejected"


_ALLOWED_APPROVAL_TRANSITIONS: Final = {
    ApprovalLifecycleState.REQUESTED: frozenset(
        {ApprovalLifecycleState.ASSIGNED, ApprovalLifecycleState.CANCELLED},
    ),
    ApprovalLifecycleState.ASSIGNED: frozenset(
        {
            ApprovalLifecycleState.IN_REVIEW,
            ApprovalLifecycleState.CANCELLED,
            ApprovalLifecycleState.EXPIRED,
        },
    ),
    ApprovalLifecycleState.IN_REVIEW: frozenset(
        {
            ApprovalLifecycleState.APPROVED,
            ApprovalLifecycleState.REJECTED,
            ApprovalLifecycleState.CANCELLED,
            ApprovalLifecycleState.EXPIRED,
        },
    ),
    ApprovalLifecycleState.APPROVED: frozenset(),
    ApprovalLifecycleState.REJECTED: frozenset(),
    ApprovalLifecycleState.CANCELLED: frozenset(),
    ApprovalLifecycleState.EXPIRED: frozenset(),
}


def _validate_canonical_approval_id(value: object, label: str) -> str:
    if type(value) is not str:
        raise ApprovalIdentityInvariantError(
            f"{label} must be str, got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise ApprovalIdentityInvariantError(
            f"{label} must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise ApprovalIdentityInvariantError(
            f"{label} must not contain leading or trailing whitespace",
        )
    if value.startswith(_DECISION_ID_PREFIX):
        raise ApprovalIdentityInvariantError(
            f"{label} must not reuse DecisionId prefix {_DECISION_ID_PREFIX!r}",
        )
    for prefix in _EXECUTION_ID_PREFIXES:
        if value.startswith(prefix):
            raise ApprovalIdentityInvariantError(
                f"{label} must not use execution identity prefix {prefix!r}",
            )
    if not value.startswith(_APPROVAL_ID_PREFIX):
        raise ApprovalIdentityInvariantError(
            f"{label} must start with {_APPROVAL_ID_PREFIX!r}",
        )
    suffix = value[len(_APPROVAL_ID_PREFIX) :]
    if not _CANONICAL_SUFFIX.fullmatch(suffix):
        raise ApprovalIdentityInvariantError(
            f"{label} suffix must match [0-9a-f]{{32}}",
        )
    return value


def validate_approval_id(value: object) -> ApprovalId:
    """Validate and return a canonical ``ApprovalId``."""
    return ApprovalId(_validate_canonical_approval_id(value, "ApprovalId"))


def mint_approval_id() -> ApprovalId:
    """Mint a new canonical ``ApprovalId``."""
    return ApprovalId(f"{_APPROVAL_ID_PREFIX}{uuid4().hex}")


def _strip_scope_field(value: object, label: str) -> str:
    if type(value) is not str:
        raise ApprovalScopeInvariantError(
            f"{label} must be str, got {type(value).__name__}",
        )
    normalized = value.strip()
    if not normalized:
        raise ApprovalScopeInvariantError(f"{label} must be non-empty")
    return normalized


def validate_approval_scope(
    *,
    tenant_id: str,
    workspace_id: str,
    decision_id: DecisionId | str,
    reference_tenant_id: str | None = None,
    reference_workspace_id: str | None = None,
    reference_decision_id: DecisionId | str | None = None,
) -> DecisionId:
    """Validate tenant/workspace scope and optional reference alignment."""
    normalized_tenant = _strip_scope_field(tenant_id, "tenant_id")
    normalized_workspace = _strip_scope_field(workspace_id, "workspace_id")
    validated_decision_id = validate_decision_id(decision_id)
    if reference_tenant_id is not None:
        reference_tenant = _strip_scope_field(
            reference_tenant_id, "reference_tenant_id"
        )
        if reference_tenant != normalized_tenant:
            raise ApprovalScopeInvariantError(
                "reference tenant_id must match approval tenant_id",
            )
    if reference_workspace_id is not None:
        reference_workspace = _strip_scope_field(
            reference_workspace_id,
            "reference_workspace_id",
        )
        if reference_workspace != normalized_workspace:
            raise ApprovalScopeInvariantError(
                "reference workspace_id must match approval workspace_id",
            )
    if reference_decision_id is not None:
        reference_decision = validate_decision_id(reference_decision_id)
        if reference_decision != validated_decision_id:
            raise ApprovalScopeInvariantError(
                "reference decision_id must match approval decision_id",
            )
    return validated_decision_id


def validate_approval_identity(
    *,
    approval_id: ApprovalId | str,
    tenant_id: str,
    workspace_id: str,
    decision_id: DecisionId | str,
    requested_by_principal_id: str,
) -> ApprovalId:
    """Validate core Approval identity fields and protect against identity reuse."""
    validated_id = validate_approval_id(approval_id)
    _strip_scope_field(tenant_id, "tenant_id")
    _strip_scope_field(workspace_id, "workspace_id")
    validate_decision_id(decision_id)
    _strip_scope_field(requested_by_principal_id, "requested_by_principal_id")
    return validated_id


def _validate_execution_provenance_ref(value: object) -> ExecutionProvenanceRef:
    if type(value) is ExecutionProvenanceRef:
        validate_task_id(value.task_id)
        validate_run_id(value.run_id)
        validate_attempt_id(value.attempt_id)
        validate_execution_id(value.execution_id)
        return value
    if isinstance(value, dict):
        required_keys = {"task_id", "run_id", "attempt_id", "execution_id"}
        if set(value.keys()) != required_keys:
            raise ApprovalReferenceInvariantError(
                "execution provenance must contain task_id, run_id, attempt_id, execution_id",
            )
        ref = ExecutionProvenanceRef(
            task_id=value["task_id"],
            run_id=value["run_id"],
            attempt_id=value["attempt_id"],
            execution_id=value["execution_id"],
        )
        validate_approval_references(execution=ref)
        return ref
    raise ApprovalReferenceInvariantError(
        "execution provenance must be ExecutionProvenanceRef",
    )


def validate_approval_references(
    *,
    references: ApprovalReferences | None = None,
    execution: ExecutionProvenanceRef | None = None,
) -> None:
    """Validate optional outbound references without owning referenced lifecycles."""
    if execution is not None:
        if type(execution) is not ExecutionProvenanceRef:
            raise ApprovalReferenceInvariantError(
                "execution provenance must be ExecutionProvenanceRef or None",
            )
        validate_task_id(execution.task_id)
        validate_run_id(execution.run_id)
        validate_attempt_id(execution.attempt_id)
        validate_execution_id(execution.execution_id)
    if references is None:
        return
    if type(references) is not ApprovalReferences:
        raise ApprovalReferenceInvariantError(
            "references must be ApprovalReferences or None",
        )
    if references.work_artifact_version_ref is not None:
        ref = references.work_artifact_version_ref
        if type(ref) is not WorkArtifactVersionRef:
            raise ApprovalReferenceInvariantError(
                "work_artifact_version_ref must be WorkArtifactVersionRef",
            )
    if references.execution is not None:
        validate_approval_references(execution=references.execution)


@dataclass(frozen=True, slots=True)
class ApprovalLifecycleTransition:
    """Explicit allowed Approval lifecycle transition."""

    from_state: ApprovalLifecycleState
    to_state: ApprovalLifecycleState


def validate_approval_transition(
    *,
    from_state: ApprovalLifecycleState,
    to_state: ApprovalLifecycleState,
) -> ApprovalLifecycleTransition:
    """Validate an architecturally legal Approval lifecycle transition."""
    if type(from_state) is not ApprovalLifecycleState:
        raise TypeError("from_state must be ApprovalLifecycleState")
    if type(to_state) is not ApprovalLifecycleState:
        raise TypeError("to_state must be ApprovalLifecycleState")
    if from_state == to_state:
        raise ApprovalLifecycleTransitionError(
            f"Unsupported Approval transition: {from_state.value} -> {to_state.value}",
        )
    allowed = _ALLOWED_APPROVAL_TRANSITIONS.get(from_state, frozenset())
    if to_state not in allowed:
        raise ApprovalLifecycleTransitionError(
            f"Unsupported Approval transition: {from_state.value} -> {to_state.value}",
        )
    return ApprovalLifecycleTransition(from_state=from_state, to_state=to_state)


class ApprovalReferences(BaseModel):
    """Neutral outbound references — no artifact or execution lifecycle ownership."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["approval_references.v1"] = SCHEMA_APPROVAL_REFERENCES_V1
    work_item_id: str | None = None
    work_artifact_version_ref: WorkArtifactVersionRef | None = None
    execution: ExecutionProvenanceRef | None = None

    @field_validator("work_item_id")
    @classmethod
    def _strip_optional_work_item_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty when provided")
        return normalized

    @field_validator("execution", mode="before")
    @classmethod
    def _validate_execution(cls, value: object) -> ExecutionProvenanceRef | None:
        if value is None:
            return None
        return _validate_execution_provenance_ref(value)

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> Self:
        return cls.model_validate_json(payload)


class HumanApprovalAction(BaseModel):
    """Immutable record of a human approval action — not authorization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["human_approval_action.v1"] = (
        SCHEMA_HUMAN_APPROVAL_ACTION_V1
    )
    approval_id: str = _NON_EMPTY
    acting_principal_id: str = _NON_EMPTY
    action: HumanApprovalActionType
    timestamp: datetime
    comment_reference: str | None = None

    @field_validator("approval_id", "acting_principal_id")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("approval_id")
    @classmethod
    def _validate_approval_id_field(cls, value: str) -> str:
        return str(validate_approval_id(value))

    @field_validator("comment_reference")
    @classmethod
    def _strip_optional_comment_reference(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("timestamp")
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> Self:
        return cls.model_validate_json(payload)


class ApprovalOutcome(BaseModel):
    """Immutable substantive approval outcome — distinct from lifecycle state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["approval_outcome.v1"] = SCHEMA_APPROVAL_OUTCOME_V1
    approval_id: str = _NON_EMPTY
    disposition: ApprovalOutcomeDisposition
    recorded_at: datetime
    summary: str | None = None

    @field_validator("approval_id")
    @classmethod
    def _validate_approval_id_field(cls, value: str) -> str:
        return str(validate_approval_id(value))

    @field_validator("summary")
    @classmethod
    def _strip_optional_summary(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("recorded_at")
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("recorded_at must be timezone-aware")
        return value

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> Self:
        return cls.model_validate_json(payload)


class ApprovalRequest(BaseModel):
    """Immutable Approval request aggregate — references Decision; owns approval lifecycle."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    schema_version: Literal["approval_request.v1"] = SCHEMA_APPROVAL_REQUEST_V1
    approval_id: str = _NON_EMPTY
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    decision_id: str = _NON_EMPTY
    requested_by_principal_id: str = _NON_EMPTY
    requested_at: datetime
    lifecycle_state: ApprovalLifecycleState = ApprovalLifecycleState.REQUESTED
    references: ApprovalReferences | None = None

    @field_validator(
        "approval_id",
        "tenant_id",
        "workspace_id",
        "decision_id",
        "requested_by_principal_id",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("approval_id")
    @classmethod
    def _validate_approval_id_field(cls, value: str) -> str:
        return str(validate_approval_id(value))

    @field_validator("decision_id")
    @classmethod
    def _validate_decision_id_field(cls, value: str) -> str:
        return str(validate_decision_id(value))

    @field_validator("requested_at")
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("requested_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_scope_and_references(self) -> ApprovalRequest:
        validate_approval_scope(
            tenant_id=self.tenant_id,
            workspace_id=self.workspace_id,
            decision_id=self.decision_id,
        )
        if self.references is not None:
            validate_approval_references(references=self.references)
            if self.references.work_artifact_version_ref is not None:
                ref = self.references.work_artifact_version_ref
                validate_approval_scope(
                    tenant_id=self.tenant_id,
                    workspace_id=self.workspace_id,
                    decision_id=self.decision_id,
                    reference_tenant_id=ref.tenant_id,
                    reference_workspace_id=ref.workspace_id,
                )
        return self

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> Self:
        return cls.model_validate_json(payload)


def approval_resource_scope(
    *,
    tenant_id: str,
    workspace_id: str,
    decision_id: DecisionId | str | None = None,
    approval_id: ApprovalId | str | None = None,
) -> str:
    """Deterministic MP-1 resource scope for Approval create or action mutations."""
    normalized_tenant = _strip_scope_field(tenant_id, "tenant_id")
    normalized_workspace = _strip_scope_field(workspace_id, "workspace_id")
    has_decision = decision_id is not None
    has_approval = approval_id is not None
    if has_decision == has_approval:
        raise ApprovalScopeInvariantError(
            "exactly one of decision_id or approval_id must be provided for approval resource scope",
        )
    if has_decision:
        validated_decision_id = validate_decision_id(decision_id)
        return (
            f"approval_request:{normalized_tenant}:{normalized_workspace}:"
            f"decision:{validated_decision_id}"
        )
    validated_approval_id = validate_approval_id(approval_id)
    return (
        f"approval_action:{normalized_tenant}:{normalized_workspace}:"
        f"approval:{validated_approval_id}"
    )


class CreateApprovalRequest(BaseModel):
    """Authoritative Approval create input for MP-4D authority-gated mutations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["create_approval_request.v1"] = (
        SCHEMA_CREATE_APPROVAL_REQUEST_V1
    )
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    decision_id: str = _NON_EMPTY
    approval_id: str = _NON_EMPTY
    acting_principal_id: str = _NON_EMPTY
    requested_by_principal_id: str | None = None
    references: ApprovalReferences | None = None
    delegator_principal_id: str | None = None
    membership: WorkspaceMembership | None = None
    membership_resolution_mode: MembershipResolutionMode = (
        MembershipResolutionMode.LOCATOR
    )
    delegation: AuthorityDelegation | None = None

    @field_validator(
        "tenant_id",
        "workspace_id",
        "decision_id",
        "approval_id",
        "acting_principal_id",
        "requested_by_principal_id",
        "delegator_principal_id",
    )
    @classmethod
    def _strip_required(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty when provided")
        return normalized

    @field_validator("approval_id")
    @classmethod
    def _validate_approval_id_field(cls, value: str) -> str:
        return str(validate_approval_id(value))

    @field_validator("decision_id")
    @classmethod
    def _validate_decision_id_field(cls, value: str) -> str:
        return str(validate_decision_id(value))

    @model_validator(mode="after")
    def _validate_scope_and_authority_locators(self) -> CreateApprovalRequest:
        validate_approval_scope(
            tenant_id=self.tenant_id,
            workspace_id=self.workspace_id,
            decision_id=self.decision_id,
        )
        if self.references is not None:
            validate_approval_references(references=self.references)
        if (
            self.membership_resolution_mode
            is MembershipResolutionMode.CANONICAL_PRINCIPAL
            and self.membership is not None
        ):
            raise ValueError(
                "canonical_principal membership resolution must not include an embedded membership locator",
            )
        if self.membership is not None:
            if self.membership.tenant_id != self.tenant_id:
                raise ValueError("membership tenant_id must match request tenant_id")
            if self.membership.workspace_id != self.workspace_id:
                raise ValueError(
                    "membership workspace_id must match request workspace_id"
                )
            if self.membership.principal_id != self.acting_principal_id:
                raise ValueError(
                    "membership principal_id must match request acting_principal_id"
                )
        if self.delegation is not None:
            if self.delegation.tenant_id != self.tenant_id:
                raise ValueError("delegation tenant_id must match request tenant_id")
            if self.delegation.workspace_id != self.workspace_id:
                raise ValueError(
                    "delegation workspace_id must match request workspace_id"
                )
            if self.delegation.delegate_principal_id != self.acting_principal_id:
                raise ValueError(
                    "delegation delegate_principal_id must match request acting_principal_id",
                )
            if (
                self.delegator_principal_id is not None
                and self.delegation.delegator_principal_id
                != self.delegator_principal_id
            ):
                raise ValueError(
                    "delegation delegator_principal_id must match request delegator_principal_id",
                )
        return self


class ExecuteHumanApprovalActionRequest(BaseModel):
    """Authoritative human approval action input for MP-4D authority-gated mutations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["execute_human_approval_action_request.v1"] = (
        SCHEMA_EXECUTE_HUMAN_APPROVAL_ACTION_REQUEST_V1
    )
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    approval_id: str = _NON_EMPTY
    acting_principal_id: str = _NON_EMPTY
    action: HumanApprovalActionType
    comment_reference: str | None = None
    delegator_principal_id: str | None = None
    membership: WorkspaceMembership | None = None
    membership_resolution_mode: MembershipResolutionMode = (
        MembershipResolutionMode.LOCATOR
    )
    delegation: AuthorityDelegation | None = None

    @field_validator(
        "tenant_id",
        "workspace_id",
        "approval_id",
        "acting_principal_id",
        "comment_reference",
        "delegator_principal_id",
    )
    @classmethod
    def _strip_required(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty when provided")
        return normalized

    @field_validator("approval_id")
    @classmethod
    def _validate_approval_id_field(cls, value: str) -> str:
        return str(validate_approval_id(value))

    @model_validator(mode="after")
    def _validate_authority_locators(self) -> ExecuteHumanApprovalActionRequest:
        if (
            self.membership_resolution_mode
            is MembershipResolutionMode.CANONICAL_PRINCIPAL
            and self.membership is not None
        ):
            raise ValueError(
                "canonical_principal membership resolution must not include an embedded membership locator",
            )
        if self.membership is not None:
            if self.membership.tenant_id != self.tenant_id:
                raise ValueError("membership tenant_id must match request tenant_id")
            if self.membership.workspace_id != self.workspace_id:
                raise ValueError(
                    "membership workspace_id must match request workspace_id"
                )
            if self.membership.principal_id != self.acting_principal_id:
                raise ValueError(
                    "membership principal_id must match request acting_principal_id"
                )
        if self.delegation is not None:
            if self.delegation.tenant_id != self.tenant_id:
                raise ValueError("delegation tenant_id must match request tenant_id")
            if self.delegation.workspace_id != self.workspace_id:
                raise ValueError(
                    "delegation workspace_id must match request workspace_id"
                )
            if self.delegation.delegate_principal_id != self.acting_principal_id:
                raise ValueError(
                    "delegation delegate_principal_id must match request acting_principal_id",
                )
            if (
                self.delegator_principal_id is not None
                and self.delegation.delegator_principal_id
                != self.delegator_principal_id
            ):
                raise ValueError(
                    "delegation delegator_principal_id must match request delegator_principal_id",
                )
        return self

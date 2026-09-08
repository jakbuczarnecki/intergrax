# © Artur Czarnecki. All rights reserved.

"""Multiplayer Decision domain contracts (MP-4B).

Neutral typed contracts for Decision identity, lifecycle, outcome, and
provenance references. Distinct from:

- ``DecisionIdentity`` / DS-CORE decision system contracts.
- ``WorkItem`` / ``WorkArtifact`` / ``WorkArtifactVersion`` — Collaborative Work.
- ``Task`` / ``Run`` / ``Attempt`` / ``Execution`` — Unified Execution Runtime.
- Approval / HITL / governance runtime — future MP-4C+ slices.

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

from intergrax.contracts.collaborative_work import WorkArtifactVersionRef
from intergrax.contracts.execution_identity import (
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.execution_provenance import ExecutionProvenanceRef

SCHEMA_DECISION_V1: Final = "decision.v1"
SCHEMA_DECISION_OUTCOME_V1: Final = "decision_outcome.v1"
SCHEMA_DECISION_REFERENCES_V1: Final = "decision_references.v1"

DecisionId = NewType("DecisionId", str)

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


class DecisionContractInvariantError(Exception):
    """Base invariant violation for MP-4B Decision contracts."""


class DecisionScopeInvariantError(DecisionContractInvariantError):
    """Tenant/workspace scope or cross-reference scope mismatch."""


class DecisionIdentityInvariantError(DecisionContractInvariantError):
    """Decision identity field validation failure."""


class DecisionLifecycleTransitionError(DecisionContractInvariantError):
    """Unsupported or invalid Decision lifecycle transition."""


class DecisionProvenanceInvariantError(DecisionContractInvariantError):
    """Invalid or inconsistent execution provenance reference."""


class DecisionLifecycleState(StrEnum):
    """Explicit collaborative Decision lifecycle — not execution or approval state."""

    DRAFT = "draft"
    PROPOSED = "proposed"
    FINALIZED = "finalized"
    SUPERSEDED = "superseded"
    CANCELLED = "cancelled"


class DecisionOutcomeDisposition(StrEnum):
    """Neutral substantive outcome disposition — not approval semantics."""

    RECORDED = "recorded"
    WITHDRAWN = "withdrawn"


_ALLOWED_DECISION_TRANSITIONS: Final = {
    DecisionLifecycleState.DRAFT: frozenset(
        {DecisionLifecycleState.PROPOSED, DecisionLifecycleState.CANCELLED},
    ),
    DecisionLifecycleState.PROPOSED: frozenset(
        {
            DecisionLifecycleState.FINALIZED,
            DecisionLifecycleState.CANCELLED,
            DecisionLifecycleState.DRAFT,
        },
    ),
    DecisionLifecycleState.FINALIZED: frozenset({DecisionLifecycleState.SUPERSEDED}),
    DecisionLifecycleState.SUPERSEDED: frozenset(),
    DecisionLifecycleState.CANCELLED: frozenset(),
}


def _validate_canonical_decision_id(value: object, label: str) -> str:
    if type(value) is not str:
        raise DecisionIdentityInvariantError(
            f"{label} must be str, got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise DecisionIdentityInvariantError(
            f"{label} must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise DecisionIdentityInvariantError(
            f"{label} must not contain leading or trailing whitespace",
        )
    for prefix in _EXECUTION_ID_PREFIXES:
        if value.startswith(prefix):
            raise DecisionIdentityInvariantError(
                f"{label} must not use execution identity prefix {prefix!r}",
            )
    if not value.startswith(_DECISION_ID_PREFIX):
        raise DecisionIdentityInvariantError(
            f"{label} must start with {_DECISION_ID_PREFIX!r}",
        )
    suffix = value[len(_DECISION_ID_PREFIX) :]
    if not _CANONICAL_SUFFIX.fullmatch(suffix):
        raise DecisionIdentityInvariantError(
            f"{label} suffix must match [0-9a-f]{{32}}",
        )
    return value


def validate_decision_id(value: object) -> DecisionId:
    """Validate and return a canonical ``DecisionId``."""
    return DecisionId(_validate_canonical_decision_id(value, "DecisionId"))


def mint_decision_id() -> DecisionId:
    """Mint a new canonical ``DecisionId``."""
    return DecisionId(f"{_DECISION_ID_PREFIX}{uuid4().hex}")


def _strip_scope_field(value: object, label: str) -> str:
    if type(value) is not str:
        raise DecisionScopeInvariantError(
            f"{label} must be str, got {type(value).__name__}",
        )
    normalized = value.strip()
    if not normalized:
        raise DecisionScopeInvariantError(f"{label} must be non-empty")
    return normalized


def validate_decision_scope(
    *,
    tenant_id: str,
    workspace_id: str,
    reference_tenant_id: str | None = None,
    reference_workspace_id: str | None = None,
) -> None:
    """Validate tenant/workspace scope and optional reference alignment."""
    normalized_tenant = _strip_scope_field(tenant_id, "tenant_id")
    normalized_workspace = _strip_scope_field(workspace_id, "workspace_id")
    if reference_tenant_id is not None:
        reference_tenant = _strip_scope_field(reference_tenant_id, "reference_tenant_id")
        if reference_tenant != normalized_tenant:
            raise DecisionScopeInvariantError(
                "reference tenant_id must match decision tenant_id",
            )
    if reference_workspace_id is not None:
        reference_workspace = _strip_scope_field(
            reference_workspace_id,
            "reference_workspace_id",
        )
        if reference_workspace != normalized_workspace:
            raise DecisionScopeInvariantError(
                "reference workspace_id must match decision workspace_id",
            )


def validate_decision_identity(
    *,
    decision_id: DecisionId | str,
    tenant_id: str,
    workspace_id: str,
    created_by_principal_id: str,
) -> DecisionId:
    """Validate core Decision identity fields."""
    validated_id = validate_decision_id(decision_id)
    _strip_scope_field(tenant_id, "tenant_id")
    _strip_scope_field(workspace_id, "workspace_id")
    _strip_scope_field(created_by_principal_id, "created_by_principal_id")
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
            raise DecisionProvenanceInvariantError(
                "execution provenance must contain task_id, run_id, attempt_id, execution_id",
            )
        ref = ExecutionProvenanceRef(
            task_id=value["task_id"],
            run_id=value["run_id"],
            attempt_id=value["attempt_id"],
            execution_id=value["execution_id"],
        )
        validate_decision_provenance(ref)
        return ref
    raise DecisionProvenanceInvariantError(
        "execution provenance must be ExecutionProvenanceRef",
    )


def validate_decision_provenance(value: ExecutionProvenanceRef | None) -> None:
    """Validate optional execution provenance without substituting UER identity."""
    if value is None:
        return
    if type(value) is not ExecutionProvenanceRef:
        raise DecisionProvenanceInvariantError(
            "execution provenance must be ExecutionProvenanceRef or None",
        )
    validate_task_id(value.task_id)
    validate_run_id(value.run_id)
    validate_attempt_id(value.attempt_id)
    validate_execution_id(value.execution_id)


@dataclass(frozen=True, slots=True)
class DecisionLifecycleTransition:
    """Explicit allowed Decision lifecycle transition."""

    from_state: DecisionLifecycleState
    to_state: DecisionLifecycleState


def validate_decision_transition(
    *,
    from_state: DecisionLifecycleState,
    to_state: DecisionLifecycleState,
) -> DecisionLifecycleTransition:
    """Validate an architecturally legal Decision lifecycle transition."""
    if type(from_state) is not DecisionLifecycleState:
        raise TypeError("from_state must be DecisionLifecycleState")
    if type(to_state) is not DecisionLifecycleState:
        raise TypeError("to_state must be DecisionLifecycleState")
    if from_state == to_state:
        raise DecisionLifecycleTransitionError(
            f"Unsupported Decision transition: {from_state.value} -> {to_state.value}",
        )
    allowed = _ALLOWED_DECISION_TRANSITIONS.get(from_state, frozenset())
    if to_state not in allowed:
        raise DecisionLifecycleTransitionError(
            f"Unsupported Decision transition: {from_state.value} -> {to_state.value}",
        )
    return DecisionLifecycleTransition(from_state=from_state, to_state=to_state)


class DecisionReferences(BaseModel):
    """Neutral outbound references — no artifact or execution lifecycle ownership."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["decision_references.v1"] = SCHEMA_DECISION_REFERENCES_V1
    work_item_id: str | None = None
    work_artifact_version_ref: WorkArtifactVersionRef | None = None

    @field_validator("work_item_id")
    @classmethod
    def _strip_optional_work_item_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty when provided")
        return normalized

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> Self:
        return cls.model_validate_json(payload)


class DecisionOutcome(BaseModel):
    """Immutable substantive outcome — distinct from lifecycle and approval state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["decision_outcome.v1"] = SCHEMA_DECISION_OUTCOME_V1
    decision_id: str = _NON_EMPTY
    outcome_version: int = Field(ge=1)
    disposition: DecisionOutcomeDisposition
    recorded_at: datetime
    summary: str | None = None

    @field_validator("decision_id")
    @classmethod
    def _validate_decision_id_field(cls, value: str) -> str:
        return str(validate_decision_id(value))

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


class Decision(BaseModel):
    """Immutable collaborative Decision aggregate — distinct from WorkArtifact."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    schema_version: Literal["decision.v1"] = SCHEMA_DECISION_V1
    decision_id: str = _NON_EMPTY
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    created_by_principal_id: str = _NON_EMPTY
    created_at: datetime
    lifecycle_state: DecisionLifecycleState = DecisionLifecycleState.DRAFT
    execution: ExecutionProvenanceRef | None = None
    references: DecisionReferences | None = None

    @field_validator(
        "decision_id",
        "tenant_id",
        "workspace_id",
        "created_by_principal_id",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("decision_id")
    @classmethod
    def _validate_decision_id_field(cls, value: str) -> str:
        return str(validate_decision_id(value))

    @field_validator("created_at")
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @field_validator("execution", mode="before")
    @classmethod
    def _validate_execution(cls, value: object) -> ExecutionProvenanceRef | None:
        if value is None:
            return None
        return _validate_execution_provenance_ref(value)

    @model_validator(mode="after")
    def _validate_scope_and_references(self) -> Decision:
        validate_decision_scope(
            tenant_id=self.tenant_id,
            workspace_id=self.workspace_id,
        )
        if self.references is not None and self.references.work_artifact_version_ref is not None:
            ref = self.references.work_artifact_version_ref
            validate_decision_scope(
                tenant_id=self.tenant_id,
                workspace_id=self.workspace_id,
                reference_tenant_id=ref.tenant_id,
                reference_workspace_id=ref.workspace_id,
            )
        validate_decision_provenance(self.execution)
        return self

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> Self:
        return cls.model_validate_json(payload)

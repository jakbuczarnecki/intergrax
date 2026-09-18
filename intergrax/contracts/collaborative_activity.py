# © Artur Czarnecki. All rights reserved.

"""Collaborative Activity & Provenance public contracts (Multiplayer MP-6A).

Semantic history of meaningful collaborative actions on the work plane —
reference-first provenance, immutable records, provider-neutral ports.

Distinct from:

- ``AgentRunTrace`` / ``TraceEvent`` — runtime observability (Plane B).
- ``RuntimeEvent`` — execution evidence spine.
- ``ProofReceipt`` — proof workload outcomes.
- ``GovernanceAuditEvent`` — governance audit channel.
- ``DecisionRecord`` (UAEP) — step-level rationale artifacts.

MP-6A freezes ownership and contract shapes; persistence ships in MP-6D+.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.collaborative_work import (
    PrincipalKind,
    WorkArtifactVersionRef,
)
from intergrax.contracts.execution_provenance import ExecutionProvenanceRef
from intergrax.contracts.governed_proof import GovernanceEvidenceRef

SCHEMA_COLLABORATIVE_ACTIVITY_V1: Final = "collaborative_activity.v1"
SCHEMA_COLLABORATIVE_ACTIVITY_ACTOR_REF_V1: Final = "collaborative_activity_actor_ref.v1"
SCHEMA_COLLABORATIVE_ACTIVITY_SCOPE_V1: Final = "collaborative_activity_scope.v1"
SCHEMA_COLLABORATIVE_ACTIVITY_OUTCOME_V1: Final = "collaborative_activity_outcome.v1"
SCHEMA_COLLABORATIVE_ACTIVITY_IDEMPOTENCY_KEY_V1: Final = (
    "collaborative_activity_idempotency_key.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_PUBLICATION_V1: Final = (
    "collaborative_activity_publication.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_QUERY_V1: Final = "collaborative_activity_query.v1"
SCHEMA_COLLABORATIVE_ACTIVITY_PAGE_CURSOR_V1: Final = (
    "collaborative_activity_page_cursor.v1"
)

_ACTIVITY_ID_PREFIX: Final = "cact_"

_NON_EMPTY = Field(min_length=1)


class CollaborativeActivitySourceDomain(StrEnum):
    """Authoritative producer domain for idempotency — not transport labels."""

    COLLABORATIVE_WORK = "collaborative_work"
    CONTEXT_VIEW = "context_view"
    COLLABORATIVE_DECISION_BINDING = "collaborative_decision_binding"
    DECISION_SYSTEM = "decision_system"
    GOVERNANCE_HITL = "governance_hitl"
    PLUGIN = "plugin"


class CollaborativeActivityType(StrEnum):
    """Frozen MP-6A taxonomy — extend only with real source capabilities."""

    WORK_ITEM_CREATED = "work_item_created"
    WORK_ITEM_UPDATED = "work_item_updated"
    WORK_ITEM_STATE_CHANGED = "work_item_state_changed"
    ASSIGNMENT_CREATED = "assignment_created"
    ASSIGNMENT_STATE_CHANGED = "assignment_state_changed"
    WORK_ARTIFACT_CREATED = "work_artifact_created"
    WORK_ARTIFACT_VERSION_PUBLISHED = "work_artifact_version_published"
    WORK_ITEM_EXECUTION_LINKED = "work_item_execution_linked"
    COLLABORATIVE_DECISION_BINDING_CREATED = "collaborative_decision_binding_created"
    DECISION_RECORDED = "decision_recorded"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESOLVED = "approval_resolved"
    CONTEXT_VIEW_COMPOSED = "context_view_composed"
    CONTEXT_VIEW_CONSUMED = "context_view_consumed"
    DELEGATION_USED = "delegation_used"
    AUTHORITY_RELEVANT_ACTION = "authority_relevant_action"
    ACTIVITY_CORRECTION = "activity_correction"


class CollaborativeActivityOutcomeStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"
    PARTIAL = "partial"


class CollaborativeActivityDurabilityClass(StrEnum):
    """Recording posture — policy binding in MP-6C+ integrations."""

    AUDIT_CRITICAL = "audit_critical"
    COLLABORATIVE = "collaborative"
    INFORMATIONAL = "informational"


class ActivityIdempotencyKey(BaseModel):
    """Canonical duplicate-delivery key — never derived from arbitrary payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_idempotency_key.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_IDEMPOTENCY_KEY_V1
    )
    source_domain: CollaborativeActivitySourceDomain
    source_stable_id: str = _NON_EMPTY
    activity_type: CollaborativeActivityType

    @field_validator("source_stable_id")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


def mint_collaborative_activity_id(*, idempotency_key: ActivityIdempotencyKey) -> str:
    """Deterministic activity identity from canonical idempotency key."""
    material = (
        f"{idempotency_key.source_domain.value}:"
        f"{idempotency_key.source_stable_id}:"
        f"{idempotency_key.activity_type.value}"
    )
    digest = hashlib.sha256(material.encode()).hexdigest()[:32]
    return f"{_ACTIVITY_ID_PREFIX}{digest}"


class CollaborativeActivityActorRef(BaseModel):
    """Canonical collaborative actor — not a display name or log label."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_actor_ref.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_ACTOR_REF_V1
    )
    tenant_id: str = _NON_EMPTY
    principal_id: str = _NON_EMPTY
    principal_kind: PrincipalKind
    delegation_id: str | None = None
    delegator_principal_id: str | None = None

    @field_validator("tenant_id", "principal_id", "delegation_id", "delegator_principal_id")
    @classmethod
    def _strip_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty when provided")
        return normalized

    @model_validator(mode="after")
    def _delegation_fields_paired(self) -> CollaborativeActivityActorRef:
        if self.delegation_id is not None and self.delegator_principal_id is None:
            raise ValueError("delegator_principal_id required when delegation_id is set")
        return self


class CollaborativeActivityScope(BaseModel):
    """Authoritative collaborative placement — work_item_id only when source-proven."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_scope.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_SCOPE_V1
    )
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    work_item_id: str | None = None

    @field_validator("tenant_id", "workspace_id", "work_item_id")
    @classmethod
    def _strip_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty when provided")
        return normalized


class WorkItemActivityTargetRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["work_item"] = "work_item"
    work_item_id: str = _NON_EMPTY

    @field_validator("work_item_id")
    @classmethod
    def _strip(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class AssignmentActivityTargetRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["assignment"] = "assignment"
    assignment_id: str = _NON_EMPTY
    work_item_id: str = _NON_EMPTY

    @field_validator("assignment_id", "work_item_id")
    @classmethod
    def _strip(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class WorkArtifactActivityTargetRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["work_artifact"] = "work_artifact"
    work_artifact_id: str = _NON_EMPTY
    work_item_id: str = _NON_EMPTY

    @field_validator("work_artifact_id", "work_item_id")
    @classmethod
    def _strip(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class WorkArtifactVersionActivityTargetRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["work_artifact_version"] = "work_artifact_version"
    version_ref: WorkArtifactVersionRef

    @model_validator(mode="after")
    def _scope_present(self) -> WorkArtifactVersionActivityTargetRef:
        _ = self.version_ref.work_item_id
        return self


class DecisionActivityTargetRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["decision"] = "decision"
    decision_id: str = _NON_EMPTY

    @field_validator("decision_id")
    @classmethod
    def _strip(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class ApprovalActivityTargetRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["approval"] = "approval"
    approval_id: str = _NON_EMPTY

    @field_validator("approval_id")
    @classmethod
    def _strip(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class ContextViewActivityTargetRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["context_view"] = "context_view"
    view_id: str = _NON_EMPTY

    @field_validator("view_id")
    @classmethod
    def _strip(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class CollaborativeDecisionBindingActivityTargetRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["collaborative_decision_binding"] = "collaborative_decision_binding"
    binding_id: str = _NON_EMPTY

    @field_validator("binding_id")
    @classmethod
    def _strip(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


CollaborativeActivityTargetRef = Annotated[
    WorkItemActivityTargetRef
    | AssignmentActivityTargetRef
    | WorkArtifactActivityTargetRef
    | WorkArtifactVersionActivityTargetRef
    | DecisionActivityTargetRef
    | ApprovalActivityTargetRef
    | ContextViewActivityTargetRef
    | CollaborativeDecisionBindingActivityTargetRef,
    Field(discriminator="kind"),
]


class ExecutionActivityProvenanceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["execution"] = "execution"
    execution: ExecutionProvenanceRef


class GovernanceEvidenceActivityProvenanceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["governance_evidence"] = "governance_evidence"
    evidence: GovernanceEvidenceRef


class ProofReceiptActivityProvenanceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["proof_receipt"] = "proof_receipt"
    proof_id: str = _NON_EMPTY

    @field_validator("proof_id")
    @classmethod
    def _strip(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class ContextViewActivityProvenanceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["context_view"] = "context_view"
    view_id: str = _NON_EMPTY

    @field_validator("view_id")
    @classmethod
    def _strip(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class DecisionActivityProvenanceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["decision"] = "decision"
    decision_id: str = _NON_EMPTY

    @field_validator("decision_id")
    @classmethod
    def _strip(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class ApprovalActivityProvenanceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["approval"] = "approval"
    approval_id: str = _NON_EMPTY

    @field_validator("approval_id")
    @classmethod
    def _strip(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class ArtifactVersionActivityProvenanceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["artifact_version"] = "artifact_version"
    version_ref: WorkArtifactVersionRef


CollaborativeActivityProvenanceRef = Annotated[
    ExecutionActivityProvenanceRef
    | GovernanceEvidenceActivityProvenanceRef
    | ProofReceiptActivityProvenanceRef
    | ContextViewActivityProvenanceRef
    | DecisionActivityProvenanceRef
    | ApprovalActivityProvenanceRef
    | ArtifactVersionActivityProvenanceRef,
    Field(discriminator="kind"),
]


class CollaborativeActivityCorrelation(BaseModel):
    """Optional cross-surface correlation — none are required on every activity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str | None = None
    step_id: str | None = None
    operation_id: str | None = None
    session_id: str | None = None
    incident_id: str | None = None

    @field_validator("run_id", "step_id", "operation_id", "session_id", "incident_id")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class CollaborativeActivityOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_outcome.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_OUTCOME_V1
    )
    status: CollaborativeActivityOutcomeStatus
    reason_code: str | None = None

    @field_validator("reason_code")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class CollaborativeActivity(BaseModel):
    """Immutable semantic collaborative action record (MP-6 core)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity.v1"] = SCHEMA_COLLABORATIVE_ACTIVITY_V1
    activity_id: str = _NON_EMPTY
    idempotency_key: ActivityIdempotencyKey
    activity_type: CollaborativeActivityType
    actor: CollaborativeActivityActorRef
    scope: CollaborativeActivityScope
    target: CollaborativeActivityTargetRef
    outcome: CollaborativeActivityOutcome
    occurred_at: datetime
    recorded_at: datetime
    provenance_refs: tuple[CollaborativeActivityProvenanceRef, ...] = ()
    correlation: CollaborativeActivityCorrelation | None = None
    caused_by_activity_id: str | None = None
    durability_class: CollaborativeActivityDurabilityClass = (
        CollaborativeActivityDurabilityClass.COLLABORATIVE
    )
    authority_delegation_id: str | None = None

    @field_validator("activity_id", "caused_by_activity_id", "authority_delegation_id")
    @classmethod
    def _strip_optional_ids(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def _actor_tenant_matches_scope(self) -> CollaborativeActivity:
        if self.actor.tenant_id != self.scope.tenant_id:
            raise ValueError("actor tenant_id must match scope tenant_id")
        return self


class CollaborativeActivityPublication(BaseModel):
    """Neutral producer command — source domains emit this, not store DTOs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_publication.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_PUBLICATION_V1
    )
    idempotency_key: ActivityIdempotencyKey
    activity_type: CollaborativeActivityType
    actor: CollaborativeActivityActorRef
    scope: CollaborativeActivityScope
    target: CollaborativeActivityTargetRef
    outcome: CollaborativeActivityOutcome
    occurred_at: datetime
    provenance_refs: tuple[CollaborativeActivityProvenanceRef, ...] = ()
    correlation: CollaborativeActivityCorrelation | None = None
    caused_by_activity_id: str | None = None
    durability_class: CollaborativeActivityDurabilityClass = (
        CollaborativeActivityDurabilityClass.COLLABORATIVE
    )
    authority_delegation_id: str | None = None

    @field_validator("caused_by_activity_id", "authority_delegation_id")
    @classmethod
    def _strip_optional_ids(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class CollaborativeActivityPageCursor(BaseModel):
    """Opaque provider-neutral pagination cursor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_page_cursor.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_PAGE_CURSOR_V1
    )
    token: str = _NON_EMPTY

    @field_validator("token")
    @classmethod
    def _strip(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class CollaborativeActivityQuery(BaseModel):
    """Authorized read intent — enforcement outside the store."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_query.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_QUERY_V1
    )
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    work_item_id: str | None = None
    actor_principal_id: str | None = None
    activity_types: tuple[CollaborativeActivityType, ...] = ()
    limit: int = Field(default=50, ge=1, le=500)
    cursor: CollaborativeActivityPageCursor | None = None
    occurred_after: datetime | None = None
    occurred_before: datetime | None = None

    @field_validator("tenant_id", "workspace_id", "work_item_id", "actor_principal_id")
    @classmethod
    def _strip_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty when provided")
        return normalized


class CollaborativeActivityPage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    activities: tuple[CollaborativeActivity, ...] = ()
    next_cursor: CollaborativeActivityPageCursor | None = None


class CollaborativeActivityPublicationPort(Protocol):
    """Source-domain facing publication seam — implemented by MP-6 ingestion."""

    def publish(self, publication: CollaborativeActivityPublication) -> CollaborativeActivity:
        """Record activity idempotently; duplicate keys return the existing record."""


class CollaborativeActivityWritePort(Protocol):
    """Domain-owned append boundary (service layer)."""

    def append(self, publication: CollaborativeActivityPublication) -> CollaborativeActivity: ...


class CollaborativeActivityReadPort(Protocol):
    """Authorized timeline query — not generic CRUD."""

    def query(self, query: CollaborativeActivityQuery) -> CollaborativeActivityPage: ...


class CollaborativeActivityAppendStore(Protocol):
    """Replaceable persistence seam (MP-6D) — append + idempotent get-by-key only."""

    def append_idempotent(
        self,
        activity: CollaborativeActivity,
    ) -> CollaborativeActivity: ...

    def get_by_idempotency_key(
        self,
        key: ActivityIdempotencyKey,
    ) -> CollaborativeActivity | None: ...

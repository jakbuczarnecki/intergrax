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

MP-6A-C1 hardens scoped idempotency identity, namespaced extensible type/source
identifiers, and event-time vs append-pagination ordering semantics.
MP-6A-C1-R1 assigns ``append_position`` and ``recorded_at`` only on materialized
``CollaborativeActivity`` at the atomic ``CollaborativeActivityAppendStore`` boundary
(producers never supply sequencing or materialization timestamps).
MP-6B freezes runtime DTO invariants (structural validation, wire symmetry, golden
identity, reference-only provenance, policy-owned effective durability).
MP-6B-C1 introduces ``CollaborativeActivityAppendIntent`` — policy-resolved durability
at the append-store boundary (store receives intent, not raw publication).
Persistence ships in MP-6D+.
"""

from __future__ import annotations

import hashlib
import re
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
SCHEMA_COLLABORATIVE_ACTIVITY_APPEND_INTENT_V1: Final = (
    "collaborative_activity_append_intent.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_QUERY_V1: Final = "collaborative_activity_query.v1"
SCHEMA_COLLABORATIVE_ACTIVITY_PAGE_CURSOR_V1: Final = (
    "collaborative_activity_page_cursor.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_TYPE_ID_V1: Final = "collaborative_activity_type_id.v1"
SCHEMA_COLLABORATIVE_ACTIVITY_SOURCE_ID_V1: Final = "collaborative_activity_source_id.v1"
SCHEMA_COLLABORATIVE_ACTIVITY_CORRELATION_V1: Final = "collaborative_activity_correlation.v1"
SCHEMA_COLLABORATIVE_ACTIVITY_PAGE_V1: Final = "collaborative_activity_page.v1"
SCHEMA_COLLABORATIVE_ACTIVITY_TARGET_WORK_ITEM_V1: Final = (
    "collaborative_activity_target.work_item.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_TARGET_ASSIGNMENT_V1: Final = (
    "collaborative_activity_target.assignment.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_TARGET_WORK_ARTIFACT_V1: Final = (
    "collaborative_activity_target.work_artifact.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_TARGET_WORK_ARTIFACT_VERSION_V1: Final = (
    "collaborative_activity_target.work_artifact_version.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_TARGET_DECISION_V1: Final = (
    "collaborative_activity_target.decision.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_TARGET_APPROVAL_V1: Final = (
    "collaborative_activity_target.approval.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_TARGET_CONTEXT_VIEW_V1: Final = (
    "collaborative_activity_target.context_view.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_TARGET_COLLABORATIVE_DECISION_BINDING_V1: Final = (
    "collaborative_activity_target.collaborative_decision_binding.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_TARGET_COLLABORATIVE_ACTIVITY_V1: Final = (
    "collaborative_activity_target.collaborative_activity.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_PROVENANCE_EXECUTION_V1: Final = (
    "collaborative_activity_provenance.execution.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_PROVENANCE_GOVERNANCE_EVIDENCE_V1: Final = (
    "collaborative_activity_provenance.governance_evidence.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_PROVENANCE_PROOF_RECEIPT_V1: Final = (
    "collaborative_activity_provenance.proof_receipt.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_PROVENANCE_CONTEXT_VIEW_V1: Final = (
    "collaborative_activity_provenance.context_view.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_PROVENANCE_DECISION_V1: Final = (
    "collaborative_activity_provenance.decision.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_PROVENANCE_APPROVAL_V1: Final = (
    "collaborative_activity_provenance.approval.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_PROVENANCE_ARTIFACT_VERSION_V1: Final = (
    "collaborative_activity_provenance.artifact_version.v1"
)

_ACTIVITY_ID_PREFIX: Final = "cact_"
_ACTIVITY_ID_HASH_SCHEME: Final = "activity-id/v1"
"""SHA-256 digest truncated to 32 hex chars — matches platform id conventions (RAG, CW bindings)."""

_RESERVED_ACTIVITY_TYPE_NAMESPACES: Final = frozenset({"intergrax", "platform"})
_RESERVED_ACTIVITY_SOURCE_NAMESPACES: Final = frozenset({"intergrax", "platform"})
_IDENTIFIER_SEGMENT_RE: Final = re.compile(r"^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$")

_NON_EMPTY = Field(min_length=1)


def _normalize_identifier_segment(value: str, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be str")
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError(f"{label} must be non-empty")
    if normalized != value.strip().lower() or value != value.strip():
        raise ValueError(f"{label} must not contain leading or trailing whitespace")
    if not _IDENTIFIER_SEGMENT_RE.fullmatch(normalized):
        raise ValueError(f"{label} must be a lowercase namespaced token segment")
    return normalized


class CollaborativeActivityTypeId(BaseModel):
    """Namespaced, plugin-extensible activity type identity — not a closed enum."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_type_id.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_TYPE_ID_V1
    )
    namespace: str = _NON_EMPTY
    name: str = _NON_EMPTY

    @field_validator("namespace", "name")
    @classmethod
    def _normalize_segments(cls, value: str, info) -> str:
        label = "namespace" if info.field_name == "namespace" else "name"
        return _normalize_identifier_segment(value, label=label)

    @property
    def qualified_id(self) -> str:
        return f"{self.namespace}.{self.name}"

    @classmethod
    def platform(cls, name: str) -> CollaborativeActivityTypeId:
        """Built-in platform taxonomy entry (reserved ``platform`` namespace)."""
        return cls(namespace="platform", name=_normalize_identifier_segment(name, label="name"))

    @classmethod
    def for_extension(cls, namespace: str, name: str) -> CollaborativeActivityTypeId:
        """Plugin-defined type — reserved platform namespaces are rejected at contract boundary."""
        ns = _normalize_identifier_segment(namespace, label="namespace")
        if ns in _RESERVED_ACTIVITY_TYPE_NAMESPACES:
            raise ValueError("extension activity types cannot use reserved namespaces")
        return cls(namespace=ns, name=_normalize_identifier_segment(name, label="name"))


class CollaborativeActivitySourceId(BaseModel):
    """Namespaced producer identity for idempotency — isolates plugins and built-ins."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_source_id.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_SOURCE_ID_V1
    )
    namespace: str = _NON_EMPTY
    name: str = _NON_EMPTY

    @field_validator("namespace", "name")
    @classmethod
    def _normalize_segments(cls, value: str, info) -> str:
        label = "namespace" if info.field_name == "namespace" else "name"
        return _normalize_identifier_segment(value, label=label)

    @property
    def qualified_id(self) -> str:
        return f"{self.namespace}.{self.name}"

    @classmethod
    def platform(cls, name: str) -> CollaborativeActivitySourceId:
        return cls(namespace="platform", name=_normalize_identifier_segment(name, label="name"))

    @classmethod
    def for_extension(cls, namespace: str, name: str) -> CollaborativeActivitySourceId:
        ns = _normalize_identifier_segment(namespace, label="namespace")
        if ns in _RESERVED_ACTIVITY_SOURCE_NAMESPACES:
            raise ValueError("extension activity sources cannot use reserved namespaces")
        return cls(namespace=ns, name=_normalize_identifier_segment(name, label="name"))


class CollaborativeActivityBuiltinType:
    """Canonical platform activity types — stable qualified IDs for integrations."""

    WORK_ITEM_CREATED = CollaborativeActivityTypeId.platform("work_item.created")
    WORK_ITEM_UPDATED = CollaborativeActivityTypeId.platform("work_item.updated")
    WORK_ITEM_STATE_CHANGED = CollaborativeActivityTypeId.platform("work_item.state_changed")
    ASSIGNMENT_CREATED = CollaborativeActivityTypeId.platform("assignment.created")
    ASSIGNMENT_STATE_CHANGED = CollaborativeActivityTypeId.platform("assignment.state_changed")
    WORK_ARTIFACT_CREATED = CollaborativeActivityTypeId.platform("work_artifact.created")
    WORK_ARTIFACT_VERSION_PUBLISHED = CollaborativeActivityTypeId.platform(
        "work_artifact.version_published"
    )
    WORK_ITEM_EXECUTION_LINKED = CollaborativeActivityTypeId.platform("work_item.execution_linked")
    COLLABORATIVE_DECISION_BINDING_CREATED = CollaborativeActivityTypeId.platform(
        "collaborative_decision_binding.created"
    )
    DECISION_RECORDED = CollaborativeActivityTypeId.platform("decision.recorded")
    APPROVAL_REQUESTED = CollaborativeActivityTypeId.platform("approval.requested")
    APPROVAL_RESOLVED = CollaborativeActivityTypeId.platform("approval.resolved")
    CONTEXT_VIEW_COMPOSED = CollaborativeActivityTypeId.platform("context_view.composed")
    CONTEXT_VIEW_CONSUMED = CollaborativeActivityTypeId.platform("context_view.consumed")
    DELEGATION_USED = CollaborativeActivityTypeId.platform("delegation.used")
    AUTHORITY_RELEVANT_ACTION = CollaborativeActivityTypeId.platform("authority.relevant_action")
    ACTIVITY_CORRECTION = CollaborativeActivityTypeId.platform("activity.correction")


class CollaborativeActivityBuiltinSource:
    """Canonical platform producer identities."""

    COLLABORATIVE_WORK = CollaborativeActivitySourceId.platform("collaborative_work")
    CONTEXT_VIEW = CollaborativeActivitySourceId.platform("context_view")
    COLLABORATIVE_DECISION_BINDING = CollaborativeActivitySourceId.platform(
        "collaborative_decision_binding"
    )
    DECISION_SYSTEM = CollaborativeActivitySourceId.platform("decision_system")
    GOVERNANCE_HITL = CollaborativeActivitySourceId.platform("governance_hitl")


class CollaborativeActivityOutcomeStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"
    PARTIAL = "partial"


class CollaborativeActivityDurabilityClass(StrEnum):
    """Recording posture for persistence/failure policy.

    Producers may **request** a class on ``CollaborativeActivityPublication``
    (``requested_durability_class``). The effective ``durability_class`` on
    materialized ``CollaborativeActivity`` is owned by MP-6C policy (may upgrade,
    never downgrade below platform minimum for the activity type).
    """

    AUDIT_CRITICAL = "audit_critical"
    COLLABORATIVE = "collaborative"
    INFORMATIONAL = "informational"


class ActivityIdempotencyKey(BaseModel):
    """Canonical duplicate-delivery key — tenant/workspace scoped; never payload-derived."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_idempotency_key.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_IDEMPOTENCY_KEY_V1
    )
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    source: CollaborativeActivitySourceId
    source_stable_id: str = _NON_EMPTY
    activity_type: CollaborativeActivityTypeId

    @field_validator("tenant_id", "workspace_id", "source_stable_id")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


def _activity_id_length_prefixed_segment(value: str) -> str:
    return f"{len(value)}:{value}"


def _activity_id_hash_material(*, idempotency_key: ActivityIdempotencyKey) -> str:
    segments = (
        _ACTIVITY_ID_HASH_SCHEME,
        _activity_id_length_prefixed_segment(idempotency_key.tenant_id),
        _activity_id_length_prefixed_segment(idempotency_key.workspace_id),
        _activity_id_length_prefixed_segment(idempotency_key.source.qualified_id),
        _activity_id_length_prefixed_segment(idempotency_key.source_stable_id),
        _activity_id_length_prefixed_segment(idempotency_key.activity_type.qualified_id),
    )
    return "|".join(segments)


def mint_collaborative_activity_id(*, idempotency_key: ActivityIdempotencyKey) -> str:
    """Deterministic activity identity from canonical scoped idempotency key."""
    material = _activity_id_hash_material(idempotency_key=idempotency_key)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
    return f"{_ACTIVITY_ID_PREFIX}{digest}"


def _scope_matches_idempotency_key(
    *,
    scope: CollaborativeActivityScope,
    idempotency_key: ActivityIdempotencyKey,
) -> None:
    if idempotency_key.tenant_id != scope.tenant_id:
        raise ValueError("idempotency_key tenant_id must match scope tenant_id")
    if idempotency_key.workspace_id != scope.workspace_id:
        raise ValueError("idempotency_key workspace_id must match scope workspace_id")


def _validate_target_scope_alignment(
    *,
    scope: CollaborativeActivityScope,
    target: CollaborativeActivityTargetRef,
) -> None:
    work_item_from_target: str | None = None
    if isinstance(target, WorkItemActivityTargetRef):
        work_item_from_target = target.work_item_id
    elif isinstance(target, AssignmentActivityTargetRef):
        work_item_from_target = target.work_item_id
    elif isinstance(target, WorkArtifactActivityTargetRef):
        work_item_from_target = target.work_item_id
    elif isinstance(target, WorkArtifactVersionActivityTargetRef):
        ref = target.version_ref
        if ref.tenant_id != scope.tenant_id:
            raise ValueError("artifact version target tenant_id must match scope tenant_id")
        if ref.workspace_id != scope.workspace_id:
            raise ValueError("artifact version target workspace_id must match scope workspace_id")
        work_item_from_target = ref.work_item_id

    if work_item_from_target is None:
        return
    if scope.work_item_id is None:
        raise ValueError("scope.work_item_id required when target references a work item")
    if scope.work_item_id != work_item_from_target:
        raise ValueError("target work_item_id must match scope work_item_id")


def _require_timezone_aware(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value


def _validate_collaborative_activity_id_value(value: str, *, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must be non-empty")
    if not normalized.startswith(_ACTIVITY_ID_PREFIX):
        raise ValueError(f"{label} must use {_ACTIVITY_ID_PREFIX} prefix")
    return normalized


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
        has_delegation = self.delegation_id is not None
        has_delegator = self.delegator_principal_id is not None
        if has_delegation != has_delegator:
            raise ValueError("delegation_id and delegator_principal_id must be set together")
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

    schema_version: Literal["collaborative_activity_target.work_item.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_TARGET_WORK_ITEM_V1
    )
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

    schema_version: Literal["collaborative_activity_target.assignment.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_TARGET_ASSIGNMENT_V1
    )
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

    schema_version: Literal["collaborative_activity_target.work_artifact.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_TARGET_WORK_ARTIFACT_V1
    )
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

    schema_version: Literal["collaborative_activity_target.work_artifact_version.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_TARGET_WORK_ARTIFACT_VERSION_V1
    )
    kind: Literal["work_artifact_version"] = "work_artifact_version"
    version_ref: WorkArtifactVersionRef

    @model_validator(mode="after")
    def _scope_present(self) -> WorkArtifactVersionActivityTargetRef:
        _ = self.version_ref.work_item_id
        return self


class DecisionActivityTargetRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_target.decision.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_TARGET_DECISION_V1
    )
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

    schema_version: Literal["collaborative_activity_target.approval.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_TARGET_APPROVAL_V1
    )
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

    schema_version: Literal["collaborative_activity_target.context_view.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_TARGET_CONTEXT_VIEW_V1
    )
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

    schema_version: Literal[
        "collaborative_activity_target.collaborative_decision_binding.v1"
    ] = SCHEMA_COLLABORATIVE_ACTIVITY_TARGET_COLLABORATIVE_DECISION_BINDING_V1
    kind: Literal["collaborative_decision_binding"] = "collaborative_decision_binding"
    binding_id: str = _NON_EMPTY

    @field_validator("binding_id")
    @classmethod
    def _strip(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class CollaborativeActivityRecordTargetRef(BaseModel):
    """Typed target for supersession/correction of an existing activity record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_target.collaborative_activity.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_TARGET_COLLABORATIVE_ACTIVITY_V1
    )
    kind: Literal["collaborative_activity"] = "collaborative_activity"
    activity_id: str = _NON_EMPTY

    @field_validator("activity_id")
    @classmethod
    def _validate_activity_id(cls, value: str) -> str:
        return _validate_collaborative_activity_id_value(value, label="activity_id")


CollaborativeActivityTargetRef = Annotated[
    WorkItemActivityTargetRef
    | AssignmentActivityTargetRef
    | WorkArtifactActivityTargetRef
    | WorkArtifactVersionActivityTargetRef
    | DecisionActivityTargetRef
    | ApprovalActivityTargetRef
    | ContextViewActivityTargetRef
    | CollaborativeDecisionBindingActivityTargetRef
    | CollaborativeActivityRecordTargetRef,
    Field(discriminator="kind"),
]


class ExecutionActivityProvenanceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_provenance.execution.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_PROVENANCE_EXECUTION_V1
    )
    kind: Literal["execution"] = "execution"
    execution: ExecutionProvenanceRef


class GovernanceEvidenceActivityProvenanceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_provenance.governance_evidence.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_PROVENANCE_GOVERNANCE_EVIDENCE_V1
    )
    kind: Literal["governance_evidence"] = "governance_evidence"
    evidence: GovernanceEvidenceRef


class ProofReceiptActivityProvenanceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_provenance.proof_receipt.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_PROVENANCE_PROOF_RECEIPT_V1
    )
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

    schema_version: Literal["collaborative_activity_provenance.context_view.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_PROVENANCE_CONTEXT_VIEW_V1
    )
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

    schema_version: Literal["collaborative_activity_provenance.decision.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_PROVENANCE_DECISION_V1
    )
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

    schema_version: Literal["collaborative_activity_provenance.approval.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_PROVENANCE_APPROVAL_V1
    )
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

    schema_version: Literal["collaborative_activity_provenance.artifact_version.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_PROVENANCE_ARTIFACT_VERSION_V1
    )
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


def _canonical_provenance_refs(
    refs: tuple[CollaborativeActivityProvenanceRef, ...],
) -> tuple[CollaborativeActivityProvenanceRef, ...]:
    """Deterministic dedupe — provenance order has no semantic chain meaning."""
    by_key: dict[str, CollaborativeActivityProvenanceRef] = {}
    for ref in refs:
        by_key[ref.model_dump_json()] = ref
    return tuple(by_key[key] for key in sorted(by_key))


def _validate_provenance_scope_alignment(
    *,
    scope: CollaborativeActivityScope,
    provenance_refs: tuple[CollaborativeActivityProvenanceRef, ...],
) -> None:
    for ref in provenance_refs:
        if isinstance(ref, ArtifactVersionActivityProvenanceRef):
            version_ref = ref.version_ref
            if version_ref.tenant_id != scope.tenant_id:
                raise ValueError("provenance artifact version tenant_id must match scope tenant_id")
            if version_ref.workspace_id != scope.workspace_id:
                raise ValueError(
                    "provenance artifact version workspace_id must match scope workspace_id"
                )
            if scope.work_item_id is not None and version_ref.work_item_id != scope.work_item_id:
                raise ValueError(
                    "provenance artifact version work_item_id must match scope work_item_id"
                )


def _validate_correction_semantics(
    *,
    activity_type: CollaborativeActivityTypeId,
    target: CollaborativeActivityTargetRef,
    caused_by_activity_id: str | None,
) -> None:
    if activity_type != CollaborativeActivityBuiltinType.ACTIVITY_CORRECTION:
        return
    has_causal = caused_by_activity_id is not None
    has_activity_target = isinstance(target, CollaborativeActivityRecordTargetRef)
    if not has_causal and not has_activity_target:
        raise ValueError(
            "activity.correction requires caused_by_activity_id or collaborative_activity target"
        )


class CollaborativeActivityCorrelation(BaseModel):
    """Optional cross-surface correlation — none are required on every activity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_correlation.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_CORRELATION_V1
    )
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

    @model_validator(mode="after")
    def _reject_fully_empty(self) -> CollaborativeActivityCorrelation:
        if (
            self.run_id is None
            and self.step_id is None
            and self.operation_id is None
            and self.session_id is None
            and self.incident_id is None
        ):
            raise ValueError(
                "correlation must include at least one identifier; use correlation=null when absent"
            )
        return self


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
    activity_type: CollaborativeActivityTypeId
    actor: CollaborativeActivityActorRef
    scope: CollaborativeActivityScope
    target: CollaborativeActivityTargetRef
    outcome: CollaborativeActivityOutcome
    occurred_at: datetime
    recorded_at: datetime = Field(
        description="Durable materialization time at append-store acceptance — not on publication",
    )
    append_position: int = Field(
        ge=1,
        description=(
            "Monotonic unique append order within (tenant_id, workspace_id) — "
            "assigned once by the configured append-store at durable materialization; "
            "not a producer input"
        ),
    )
    provenance_refs: tuple[CollaborativeActivityProvenanceRef, ...] = ()
    correlation: CollaborativeActivityCorrelation | None = None
    caused_by_activity_id: str | None = None
    durability_class: CollaborativeActivityDurabilityClass = Field(
        default=CollaborativeActivityDurabilityClass.COLLABORATIVE,
        description="Effective durability assigned by MP-6C policy at materialization",
    )

    @field_validator("activity_id")
    @classmethod
    def _validate_activity_id(cls, value: str) -> str:
        return _validate_collaborative_activity_id_value(value, label="activity_id")

    @field_validator("caused_by_activity_id")
    @classmethod
    def _validate_caused_by(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_collaborative_activity_id_value(value, label="caused_by_activity_id")

    @field_validator("occurred_at", "recorded_at")
    @classmethod
    def _validate_datetimes(cls, value: datetime, info) -> datetime:
        return _require_timezone_aware(value, label=str(info.field_name))

    @field_validator("provenance_refs", mode="after")
    @classmethod
    def _canonicalize_provenance(
        cls,
        value: tuple[CollaborativeActivityProvenanceRef, ...],
    ) -> tuple[CollaborativeActivityProvenanceRef, ...]:
        return _canonical_provenance_refs(value)

    @model_validator(mode="after")
    def _alignment_invariants(self) -> CollaborativeActivity:
        if self.actor.tenant_id != self.scope.tenant_id:
            raise ValueError("actor tenant_id must match scope tenant_id")
        _scope_matches_idempotency_key(scope=self.scope, idempotency_key=self.idempotency_key)
        if self.idempotency_key.activity_type != self.activity_type:
            raise ValueError("activity_type must match idempotency_key.activity_type")
        expected_id = mint_collaborative_activity_id(idempotency_key=self.idempotency_key)
        if self.activity_id != expected_id:
            raise ValueError("activity_id must equal mint_collaborative_activity_id(idempotency_key)")
        _validate_target_scope_alignment(scope=self.scope, target=self.target)
        _validate_provenance_scope_alignment(scope=self.scope, provenance_refs=self.provenance_refs)
        _validate_correction_semantics(
            activity_type=self.idempotency_key.activity_type,
            target=self.target,
            caused_by_activity_id=self.caused_by_activity_id,
        )
        if (
            self.caused_by_activity_id is not None
            and self.caused_by_activity_id == self.activity_id
        ):
            raise ValueError("caused_by_activity_id must not equal activity_id")
        return self


class CollaborativeActivityPublication(BaseModel):
    """Neutral producer command — source domains emit this, not store DTOs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_publication.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_PUBLICATION_V1
    )
    idempotency_key: ActivityIdempotencyKey
    actor: CollaborativeActivityActorRef
    scope: CollaborativeActivityScope
    target: CollaborativeActivityTargetRef
    outcome: CollaborativeActivityOutcome
    occurred_at: datetime
    provenance_refs: tuple[CollaborativeActivityProvenanceRef, ...] = ()
    correlation: CollaborativeActivityCorrelation | None = None
    caused_by_activity_id: str | None = None
    requested_durability_class: CollaborativeActivityDurabilityClass = Field(
        default=CollaborativeActivityDurabilityClass.COLLABORATIVE,
        description="Producer suggestion only — effective durability is policy-owned (MP-6C)",
    )

    @property
    def activity_type(self) -> CollaborativeActivityTypeId:
        """Semantic activity kind — authoritative copy lives on the idempotency key."""
        return self.idempotency_key.activity_type

    @field_validator("caused_by_activity_id")
    @classmethod
    def _validate_caused_by(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_collaborative_activity_id_value(value, label="caused_by_activity_id")

    @field_validator("occurred_at")
    @classmethod
    def _validate_occurred_at(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value, label="occurred_at")

    @field_validator("provenance_refs", mode="after")
    @classmethod
    def _canonicalize_provenance(
        cls,
        value: tuple[CollaborativeActivityProvenanceRef, ...],
    ) -> tuple[CollaborativeActivityProvenanceRef, ...]:
        return _canonical_provenance_refs(value)

    @model_validator(mode="after")
    def _alignment_invariants(self) -> CollaborativeActivityPublication:
        if self.actor.tenant_id != self.scope.tenant_id:
            raise ValueError("actor tenant_id must match scope tenant_id")
        _scope_matches_idempotency_key(scope=self.scope, idempotency_key=self.idempotency_key)
        _validate_target_scope_alignment(scope=self.scope, target=self.target)
        _validate_provenance_scope_alignment(scope=self.scope, provenance_refs=self.provenance_refs)
        _validate_correction_semantics(
            activity_type=self.idempotency_key.activity_type,
            target=self.target,
            caused_by_activity_id=self.caused_by_activity_id,
        )
        return self


class CollaborativeActivityAppendIntent(BaseModel):
    """Policy-validated append command — store-facing persistence input (MP-6B-C1).

    Produced by the MP-6 policy / ingestion layer after semantic validation.
    Wraps the producer ``CollaborativeActivityPublication`` and carries the
    platform-resolved ``effective_durability_class`` authoritative for
    materialized ``CollaborativeActivity.durability_class``.

    Not a producer DTO, materialized record, or policy object.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_append_intent.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_APPEND_INTENT_V1
    )
    publication: CollaborativeActivityPublication
    effective_durability_class: CollaborativeActivityDurabilityClass = Field(
        description="Policy-resolved durability — authoritative for persistence",
    )


class CollaborativeActivityPageCursor(BaseModel):
    """Opaque provider-neutral continuation token (append/snapshot position — not event time)."""

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
    """Read intent shape — caller/service must supply authority-validated scope (MP-6E).

    ``cursor`` is valid only for the same canonical filter scope (tenant, workspace,
    filters, and sort contract) that produced it — not reusable across different queries.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_query.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_QUERY_V1
    )
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    work_item_id: str | None = None
    actor_principal_id: str | None = None
    activity_types: tuple[CollaborativeActivityTypeId, ...] = ()
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

    @field_validator("occurred_after", "occurred_before")
    @classmethod
    def _validate_bounds(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _require_timezone_aware(value, label="occurred bound")

    @field_validator("activity_types", mode="after")
    @classmethod
    def _dedupe_activity_types(
        cls,
        value: tuple[CollaborativeActivityTypeId, ...],
    ) -> tuple[CollaborativeActivityTypeId, ...]:
        if not value:
            return value
        seen: set[str] = set()
        deduped: list[CollaborativeActivityTypeId] = []
        for activity_type in value:
            qualified = activity_type.qualified_id
            if qualified in seen:
                continue
            seen.add(qualified)
            deduped.append(activity_type)
        return tuple(deduped)

    @model_validator(mode="after")
    def _query_invariants(self) -> CollaborativeActivityQuery:
        if (
            self.occurred_after is not None
            and self.occurred_before is not None
            and self.occurred_after > self.occurred_before
        ):
            raise ValueError("occurred_after must be <= occurred_before")
        return self


class CollaborativeActivityPage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_page.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_PAGE_V1
    )
    activities: tuple[CollaborativeActivity, ...] = ()
    next_cursor: CollaborativeActivityPageCursor | None = None


class CollaborativeActivityPublicationPort(Protocol):
    """Producer-facing ingress — source domains publish; MP-6 ingestion implements."""

    def publish(self, publication: CollaborativeActivityPublication) -> CollaborativeActivity:
        """Record activity idempotently; duplicate keys return the existing record."""
        ...


class CollaborativeActivityWritePort(Protocol):
    """Internal MP-6 service boundary — producer publication ingress (MP-6C).

    Implementations resolve ingestion policy (including effective durability),
    build ``CollaborativeActivityAppendIntent``, and delegate durable sequencing
    and materialization timestamps to ``CollaborativeActivityAppendStore``.
    Callers must not pre-build materialized ``CollaborativeActivity`` records.
    """

    def append(self, publication: CollaborativeActivityPublication) -> CollaborativeActivity: ...


class CollaborativeActivityReadPort(Protocol):
    """Timeline query port — authority resolution is caller/service responsibility (MP-6E)."""

    def query(self, query: CollaborativeActivityQuery) -> CollaborativeActivityPage: ...


class CollaborativeActivityAppendStore(Protocol):
    """Replaceable persistence seam (MP-6D) — atomic append materialization.

    ``append_idempotent`` atomically resolves duplicate idempotency keys,
    assigns workspace ``append_position`` for new activities only,
    assigns ``recorded_at`` at durable acceptance, and persists the
    materialized ``CollaborativeActivity``. Duplicate keys return the
    original materialized record without allocating a new position.

    Implementations must not require a service-level check-then-insert
    sequence; idempotency, position allocation, timestamp assignment, and
    durable append form one consistent semantic operation.

    Policy (authority, namespace authorization, target semantics, effective
    durability) is out of scope — validated ``CollaborativeActivityAppendIntent``
    is the append input; the store owns persistence semantics only and must
    materialize ``durability_class`` from ``intent.effective_durability_class``
    (never from ``intent.publication.requested_durability_class``). Duplicate
    idempotency keys return the original materialized record without mutating
    historical ``durability_class``.
    """

    def append_idempotent(
        self,
        intent: CollaborativeActivityAppendIntent,
    ) -> CollaborativeActivity: ...

    def get_by_idempotency_key(
        self,
        key: ActivityIdempotencyKey,
    ) -> CollaborativeActivity | None: ...

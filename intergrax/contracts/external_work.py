# © Artur Czarnecki. All rights reserved.

"""Provider-neutral external work domain contracts (GEC-1 / GEC-2 / platform).

Composes existing Intergrax identity, HITL, interrupt, digest, and money primitives.
GEC-2 adds the interaction model (request/snapshot/timeline/evidence/capabilities)
consumed by the provider-neutral integration boundary.

Does not own transport implementations, adapter lifecycle (GEC-3), HITL UX (GEC-4),
policy gates (GEC-5), or receipts (GEC-6).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Final, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from intergrax.contracts.actor_identity import ActorIdentity
from intergrax.contracts.money import MoneyAmount
from intergrax.contracts.validation import ValidationResult

# Same digest shape as hosted profile digests (``sha256:<64 lowercase hex>``).
_CONTENT_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_NON_EMPTY_ID = Field(min_length=1)

SCHEMA_COMMERCIAL_QUOTE_V1: Final = "commercial_quote.v1"
SCHEMA_QUOTE_ACCEPTANCE_V1: Final = "quote_acceptance_evidence.v1"
SCHEMA_EXTERNAL_DELIVERABLE_V1: Final = "external_deliverable_ref.v1"
SCHEMA_EXTERNAL_TASK_CORRELATION_V1: Final = "external_task_correlation.v1"
SCHEMA_EXTERNAL_CONTRACTOR_IDENTITY_V1: Final = "external_contractor_identity.v1"
SCHEMA_EXTERNAL_WORK_CREATE_REQUEST_V1: Final = "external_work_create_request.v1"
SCHEMA_EXTERNAL_WORK_SNAPSHOT_V1: Final = "external_work_snapshot.v1"
SCHEMA_EXTERNAL_WORK_TIMELINE_EVENT_V1: Final = "external_work_timeline_event.v1"
SCHEMA_EXTERNAL_PROVIDER_EVIDENCE_REF_V1: Final = "external_provider_evidence_ref.v1"
SCHEMA_EXTERNAL_WORK_PROVIDER_DESCRIPTOR_V1: Final = "external_work_provider_descriptor.v1"


def validate_content_digest(value: str) -> str:
    """Validate a content digest string using the platform ``sha256:`` convention."""
    normalized = value.strip()
    if not _CONTENT_DIGEST_RE.match(normalized):
        raise ValueError("digest must match sha256:<64 lowercase hex>")
    return normalized


def _require_aware_utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


class ExternalWorkStatus(StrEnum):
    """Normalized external-work progress — distinct from Nexus ``TaskState``.

    Commercial/quote stages are external-work concerns and must not inflate
    the global Nexus task lifecycle.
    """

    CREATED = "created"
    INITIALIZING = "initializing"
    QUOTE_PENDING = "quote_pending"
    QUOTE_AVAILABLE = "quote_available"
    WAITING_FOR_ACCEPTANCE = "waiting_for_acceptance"
    ACCEPTED = "accepted"
    EXECUTING = "executing"
    WAITING_FOR_HUMAN = "waiting_for_human"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


TERMINAL_EXTERNAL_WORK_STATUSES: frozenset[ExternalWorkStatus] = frozenset(
    {
        ExternalWorkStatus.COMPLETED,
        ExternalWorkStatus.FAILED,
        ExternalWorkStatus.CANCELLED,
        ExternalWorkStatus.EXPIRED,
    }
)


def is_terminal_external_work_status(status: ExternalWorkStatus) -> bool:
    """Return True when no further external-work progress is expected."""
    return status in TERMINAL_EXTERNAL_WORK_STATUSES


class QuoteLifecycleState(StrEnum):
    """Lifecycle of a commercial quote offer (not Nexus task state)."""

    DRAFT = "draft"
    OFFERED = "offered"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"


class ExternalContractorIdentity(BaseModel):
    """Identity of an external contractor product/agent — no HTTP URL required.

    Composes integration ``provider_id`` (catalog slug) with external ids.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["external_contractor_identity.v1"] = (
        SCHEMA_EXTERNAL_CONTRACTOR_IDENTITY_V1
    )
    provider_id: str = _NON_EMPTY_ID
    contractor_id: str = _NON_EMPTY_ID
    external_agent_id: str | None = None
    display_label: str | None = None
    protocol_id: str = _NON_EMPTY_ID
    descriptor_ref: str | None = None
    descriptor_digest: str | None = None

    @field_validator("provider_id", "contractor_id", "protocol_id")
    @classmethod
    def _strip_required_ids(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("identifier must be non-empty")
        return normalized

    @field_validator("external_agent_id", "display_label", "descriptor_ref")
    @classmethod
    def _strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("descriptor_digest")
    @classmethod
    def _validate_optional_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_content_digest(value)


class ExternalTaskCorrelation(BaseModel):
    """Stable mapping Intergrax task/run ↔ external contractor task.

    Invariant: ``external_task_id`` never replaces Intergrax ``task_id``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["external_task_correlation.v1"] = (
        SCHEMA_EXTERNAL_TASK_CORRELATION_V1
    )
    task_id: str = _NON_EMPTY_ID
    run_id: str | None = None
    correlation_id: str | None = None
    provider_id: str = _NON_EMPTY_ID
    external_task_id: str = _NON_EMPTY_ID
    idempotency_key: str | None = None

    @field_validator("task_id", "provider_id", "external_task_id")
    @classmethod
    def _strip_required_ids(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("identifier must be non-empty")
        return normalized

    @field_validator("run_id", "correlation_id", "idempotency_key")
    @classmethod
    def _strip_optional_ids(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class CommercialQuote(BaseModel):
    """Immutable commercial quote bound to correlated Intergrax/external task ids."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["commercial_quote.v1"] = SCHEMA_COMMERCIAL_QUOTE_V1
    quote_id: str = _NON_EMPTY_ID
    task_id: str = _NON_EMPTY_ID
    run_id: str | None = None
    external_task_id: str = _NON_EMPTY_ID
    provider_id: str = _NON_EMPTY_ID
    version: int = Field(ge=1)
    amount: MoneyAmount
    scope_description: str = Field(min_length=1)
    scope_digest: str = Field(min_length=1)
    created_at: datetime
    expires_at: datetime | None = None
    lifecycle_state: QuoteLifecycleState = QuoteLifecycleState.OFFERED
    extension: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator(
        "quote_id",
        "task_id",
        "external_task_id",
        "provider_id",
        "scope_description",
    )
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field must be non-empty")
        return normalized

    @field_validator("run_id")
    @classmethod
    def _strip_optional_run_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("scope_digest")
    @classmethod
    def _validate_scope_digest(cls, value: str) -> str:
        return validate_content_digest(value)

    @field_validator("created_at", "expires_at")
    @classmethod
    def _aware_timestamps(
        cls, value: datetime | None, info: ValidationInfo
    ) -> datetime | None:
        if value is None:
            return None
        return _require_aware_utc(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def _expiration_after_creation(self) -> CommercialQuote:
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("expires_at must be later than created_at")
        return self

    def is_expired_at(self, evaluation_time: datetime) -> bool:
        """Return True when ``evaluation_time`` is at or after ``expires_at``."""
        aware = _require_aware_utc(evaluation_time, field_name="evaluation_time")
        if self.expires_at is None:
            return False
        return aware >= self.expires_at


class QuoteAcceptanceEvidence(BaseModel):
    """Acceptance evidence only — does not authorize, pay, or mutate lifecycle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["quote_acceptance_evidence.v1"] = SCHEMA_QUOTE_ACCEPTANCE_V1
    acceptance_id: str = _NON_EMPTY_ID
    quote_id: str = _NON_EMPTY_ID
    quote_version: int = Field(ge=1)
    scope_digest: str = Field(min_length=1)
    actor: ActorIdentity
    accepted_at: datetime
    hitl_decision_id: str | None = None
    interrupt_id: str | None = None
    policy_decision_ref: str | None = None

    @field_validator("acceptance_id", "quote_id")
    @classmethod
    def _strip_required_ids(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("identifier must be non-empty")
        return normalized

    @field_validator("scope_digest")
    @classmethod
    def _validate_scope_digest(cls, value: str) -> str:
        return validate_content_digest(value)

    @field_validator("accepted_at")
    @classmethod
    def _aware_accepted_at(cls, value: datetime) -> datetime:
        return _require_aware_utc(value, field_name="accepted_at")

    @field_validator("hitl_decision_id", "interrupt_id", "policy_decision_ref")
    @classmethod
    def _strip_optional_refs(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class ExternalDeliverableRef(BaseModel):
    """Workspace-safe deliverable pointer — not a local-file or public-URL assumption.

    Composes the same digest/size conventions as ``ApplicationArtifactRef`` without
    requiring harness artifact provenance (external deliverables arrive pre-store).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["external_deliverable_ref.v1"] = SCHEMA_EXTERNAL_DELIVERABLE_V1
    deliverable_id: str = _NON_EMPTY_ID
    task_id: str = _NON_EMPTY_ID
    external_task_id: str | None = None
    kind: str = _NON_EMPTY_ID
    media_type: str = Field(default="application/octet-stream", min_length=1)
    resource_uri: str = Field(
        min_length=1,
        description="Workspace-safe resource reference (scheme-agnostic; not a public URL requirement).",
    )
    content_digest: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    created_at: datetime
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("deliverable_id", "task_id", "kind", "media_type", "resource_uri")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field must be non-empty")
        return normalized

    @field_validator("external_task_id")
    @classmethod
    def _strip_optional_external_task(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("content_digest")
    @classmethod
    def _validate_optional_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_content_digest(value)

    @field_validator("created_at")
    @classmethod
    def _aware_created_at(cls, value: datetime) -> datetime:
        return _require_aware_utc(value, field_name="created_at")


def validate_quote_acceptance_match(
    quote: CommercialQuote,
    acceptance: QuoteAcceptanceEvidence,
    *,
    evaluation_time: datetime,
) -> ValidationResult:
    """Pure quote/acceptance matching — no authz, policy, wallet, or task checks."""
    errors: list[str] = []
    if acceptance.quote_id != quote.quote_id:
        errors.append("quote_id mismatch")
    if acceptance.quote_version != quote.version:
        errors.append("quote_version mismatch")
    if acceptance.scope_digest != quote.scope_digest:
        errors.append("scope_digest mismatch")
    if quote.is_expired_at(evaluation_time):
        errors.append("quote expired at evaluation_time")
    return ValidationResult(valid=not errors, errors=errors)


class ExternalWorkCapability(StrEnum):
    """Reusable external-work feature tokens — not A2A skill names.

    Distinct from ``PlatformIntegrationCapability`` (connect/read/write/health).
    """

    QUOTE_FIRST = "quote_first"
    HUMAN_CONTINUATION = "human_continuation"
    CANCELLATION = "cancellation"
    TIMELINE = "timeline"
    DELIVERABLES = "deliverables"
    EVIDENCE_REFS = "evidence_refs"
    ASYNC_EXECUTION = "async_execution"


class ExternalWorkErrorCode(StrEnum):
    """Structured error codes for the external-work integration boundary."""

    PROVIDER_UNAVAILABLE = "provider_unavailable"
    AUTH_CONFIG_FAILED = "auth_config_failed"
    INVALID_REQUEST = "invalid_request"
    TASK_NOT_FOUND = "task_not_found"
    QUOTE_UNAVAILABLE = "quote_unavailable"
    ACCEPTANCE_CONFLICT = "acceptance_conflict"
    QUOTE_CHANGED_OR_EXPIRED = "quote_changed_or_expired"
    OPERATION_NOT_SUPPORTED = "operation_not_supported"
    CANCELLATION_REJECTED = "cancellation_rejected"
    TRANSIENT_REMOTE_FAILURE = "transient_remote_failure"
    PERMANENT_PROVIDER_FAILURE = "permanent_provider_failure"
    MALFORMED_PROVIDER_RESPONSE = "malformed_provider_response"


# Codes that callers may retry without inventing new middleware.
RETRYABLE_EXTERNAL_WORK_ERROR_CODES: frozenset[ExternalWorkErrorCode] = frozenset(
    {
        ExternalWorkErrorCode.PROVIDER_UNAVAILABLE,
        ExternalWorkErrorCode.TRANSIENT_REMOTE_FAILURE,
    }
)


def is_retryable_external_work_error(code: ExternalWorkErrorCode) -> bool:
    """Return True when the error is classified as transient/retryable."""
    return code in RETRYABLE_EXTERNAL_WORK_ERROR_CODES


class ExternalProviderEvidenceKind(StrEnum):
    """Kinds of provider-supplied evidence references (not Intergrax proof)."""

    TASK_EVENT = "task_event"
    RECEIPT = "receipt"
    TOOL_LOG = "tool_log"
    RESULT = "result"
    OTHER = "other"


class ExternalWorkProviderDescriptor(BaseModel):
    """Discovery result for an external-work provider — not an A2A Agent Card copy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["external_work_provider_descriptor.v1"] = (
        SCHEMA_EXTERNAL_WORK_PROVIDER_DESCRIPTOR_V1
    )
    identity: ExternalContractorIdentity
    capabilities: tuple[ExternalWorkCapability, ...] = ()
    protocol_id: str = _NON_EMPTY_ID
    descriptor_digest: str | None = None
    schema_id: str | None = None

    @field_validator("protocol_id")
    @classmethod
    def _strip_protocol_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("protocol_id must be non-empty")
        return normalized

    @field_validator("descriptor_digest")
    @classmethod
    def _validate_optional_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_content_digest(value)

    @field_validator("schema_id")
    @classmethod
    def _strip_optional_schema_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    def supports(self, capability: ExternalWorkCapability) -> bool:
        """Return True when the descriptor advertises ``capability``."""
        return capability in self.capabilities


class ExternalWorkCreateRequest(BaseModel):
    """Thin create/correlate request composed from platform references.

    Does not duplicate Nexus task models, tenant context, or policy payloads.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["external_work_create_request.v1"] = (
        SCHEMA_EXTERNAL_WORK_CREATE_REQUEST_V1
    )
    provider_id: str = _NON_EMPTY_ID
    task_id: str = _NON_EMPTY_ID
    run_id: str | None = None
    correlation_id: str | None = None
    requested_capability: str = _NON_EMPTY_ID
    scope_description: str = Field(min_length=1)
    scope_digest: str = Field(min_length=1)
    idempotency_key: str = _NON_EMPTY_ID
    workspace_ref: str | None = None
    budget_limit: MoneyAmount | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator(
        "provider_id",
        "task_id",
        "requested_capability",
        "scope_description",
        "idempotency_key",
    )
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field must be non-empty")
        return normalized

    @field_validator("run_id", "correlation_id", "workspace_ref")
    @classmethod
    def _strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("scope_digest")
    @classmethod
    def _validate_scope_digest(cls, value: str) -> str:
        return validate_content_digest(value)


class ExternalWorkSnapshot(BaseModel):
    """Canonical current-state view of external work — not arbitrary provider JSON."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["external_work_snapshot.v1"] = SCHEMA_EXTERNAL_WORK_SNAPSHOT_V1
    correlation: ExternalTaskCorrelation
    status: ExternalWorkStatus
    quote: CommercialQuote | None = None
    created_at: datetime
    updated_at: datetime
    provider_state_label: str | None = None
    failure_ref: str | None = None
    deliverable_count: int | None = Field(default=None, ge=0)
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("created_at", "updated_at")
    @classmethod
    def _aware_timestamps(cls, value: datetime, info: ValidationInfo) -> datetime:
        return _require_aware_utc(value, field_name=str(info.field_name))

    @field_validator("provider_state_label", "failure_ref")
    @classmethod
    def _strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def _updated_not_before_created(self) -> ExternalWorkSnapshot:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be >= created_at")
        return self

    @property
    def is_terminal(self) -> bool:
        """True when ``status`` is a terminal external-work status."""
        return is_terminal_external_work_status(self.status)


class ExternalWorkTimelineEvent(BaseModel):
    """Provider-observed fact — not an Intergrax runtime trace event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["external_work_timeline_event.v1"] = (
        SCHEMA_EXTERNAL_WORK_TIMELINE_EVENT_V1
    )
    event_id: str = _NON_EMPTY_ID
    task_id: str = _NON_EMPTY_ID
    external_task_id: str = _NON_EMPTY_ID
    provider_id: str = _NON_EMPTY_ID
    event_kind: str = _NON_EMPTY_ID
    status: ExternalWorkStatus | None = None
    provider_timestamp: datetime
    ingested_at: datetime | None = None
    summary: str = ""
    evidence_ref: str | None = None
    provider_sequence: int | None = Field(default=None, ge=0)
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator(
        "event_id",
        "task_id",
        "external_task_id",
        "provider_id",
        "event_kind",
    )
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field must be non-empty")
        return normalized

    @field_validator("provider_timestamp", "ingested_at")
    @classmethod
    def _aware_timestamps(
        cls, value: datetime | None, info: ValidationInfo
    ) -> datetime | None:
        if value is None:
            return None
        return _require_aware_utc(value, field_name=str(info.field_name))

    @field_validator("evidence_ref")
    @classmethod
    def _strip_optional_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class ExternalProviderEvidenceRef(BaseModel):
    """Provider-supplied evidence pointer — distinct from Intergrax ProofReceipt.

    GEC-2 exposes references only. Signing, persistence, and verification are deferred.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["external_provider_evidence_ref.v1"] = (
        SCHEMA_EXTERNAL_PROVIDER_EVIDENCE_REF_V1
    )
    evidence_id: str = _NON_EMPTY_ID
    task_id: str = _NON_EMPTY_ID
    external_task_id: str = _NON_EMPTY_ID
    provider_id: str = _NON_EMPTY_ID
    kind: ExternalProviderEvidenceKind
    resource_uri: str = Field(
        min_length=1,
        description="Provider-local evidence reference (scheme-agnostic; not a proof claim).",
    )
    content_digest: str | None = None
    created_at: datetime
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator(
        "evidence_id",
        "task_id",
        "external_task_id",
        "provider_id",
        "resource_uri",
    )
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field must be non-empty")
        return normalized

    @field_validator("content_digest")
    @classmethod
    def _validate_optional_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_content_digest(value)

    @field_validator("created_at")
    @classmethod
    def _aware_created_at(cls, value: datetime) -> datetime:
        return _require_aware_utc(value, field_name="created_at")

# © Artur Czarnecki. All rights reserved.

"""Provider-neutral external work domain contracts (GEC-1 / platform).

Composes existing Intergrax identity, HITL, interrupt, digest, and money primitives.
Does not own transport (GEC-2), adapter lifecycle (GEC-3), HITL UX (GEC-4),
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

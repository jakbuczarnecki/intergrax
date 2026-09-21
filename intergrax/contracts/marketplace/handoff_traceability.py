# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Discovery → selection → handoff traceability contracts (ME-10)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.release_identity import (
    CapabilityReleaseIdentity,
)
from intergrax.contracts.marketplace.query_context import MarketplaceQueryContext

SCHEMA_CAPABILITY_DISCOVERY_TRACE_FACTS_V1: Final = (
    "capability_discovery_trace_facts.v1"
)
SCHEMA_CAPABILITY_MARKETPLACE_EXPLICIT_SELECTION_V1: Final = (
    "capability_marketplace_explicit_selection.v1"
)
SCHEMA_CAPABILITY_HANDOFF_ENVELOPE_V1: Final = "capability_handoff_envelope.v1"


class CapabilityHandoffConsumerTarget(StrEnum):
    """Downstream bounded context — not an execution grant."""

    AGENT_DOMAIN = "agent_domain"
    TOOL_DOMAIN = "tool_domain"
    SKILL_DOMAIN = "skill_domain"


def consumer_target_for_kind(kind: CapabilityKind) -> CapabilityHandoffConsumerTarget:
    if kind is CapabilityKind.AGENT:
        return CapabilityHandoffConsumerTarget.AGENT_DOMAIN
    if kind is CapabilityKind.TOOL:
        return CapabilityHandoffConsumerTarget.TOOL_DOMAIN
    if kind is CapabilityKind.SKILL:
        return CapabilityHandoffConsumerTarget.SKILL_DOMAIN
    raise ValueError(f"unsupported capability kind for handoff target: {kind.value}")


class CapabilityDiscoveryTraceFacts(BaseModel):
    """Observational discovery pipeline summary — no full candidate payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_discovery_trace_facts.v1"] = (
        SCHEMA_CAPABILITY_DISCOVERY_TRACE_FACTS_V1
    )
    discovery_correlation_id: str = Field(min_length=1)
    query_correlation_id: str | None = None
    marketplace_query_context: MarketplaceQueryContext
    visible_candidate_count: int = Field(ge=0)
    governed_admissible_count: int = Field(ge=0)
    ranking_strategy_id: str | None = None
    governance_evaluator_ids: tuple[str, ...] = ()

    @field_validator("discovery_correlation_id")
    @classmethod
    def _validate_discovery_correlation_id(cls, value: str) -> str:
        return require_non_empty_text(value, label="discovery_correlation_id")

    @model_validator(mode="after")
    def _counts_consistent(self) -> CapabilityDiscoveryTraceFacts:
        if self.governed_admissible_count > self.visible_candidate_count:
            raise ValueError(
                "governed_admissible_count cannot exceed visible_candidate_count",
            )
        return self


class CapabilityMarketplaceExplicitSelection(BaseModel):
    """Explicit governed selection — distinct from ranking or recommendation output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_marketplace_explicit_selection.v1"] = (
        SCHEMA_CAPABILITY_MARKETPLACE_EXPLICIT_SELECTION_V1
    )
    selection_id: str = Field(min_length=1)
    discovery_correlation_id: str = Field(min_length=1)
    selected_release: CapabilityReleaseIdentity
    selector_id: str = Field(min_length=1)
    listing_id: str | None = None
    governance_evidence_ref: str | None = None
    ranking_evidence_ref: str | None = None

    @field_validator("selection_id", "discovery_correlation_id", "selector_id")
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return require_non_empty_text(value, label=str(info.field_name))


class CapabilityHandoffEnvelope(BaseModel):
    """Neutral capability handoff — ends before domain execution or usage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_handoff_envelope.v1"] = (
        SCHEMA_CAPABILITY_HANDOFF_ENVELOPE_V1
    )
    handoff_id: str = Field(min_length=1)
    tenant_id: str | None = None
    selected_release: CapabilityReleaseIdentity
    discovery_correlation_id: str = Field(min_length=1)
    selection_id: str = Field(min_length=1)
    consumer_target: CapabilityHandoffConsumerTarget
    downstream_consumer_id: str = Field(min_length=1)
    discovery_trace: CapabilityDiscoveryTraceFacts
    explicit_selection: CapabilityMarketplaceExplicitSelection
    recorded_at: datetime

    @field_validator(
        "handoff_id",
        "discovery_correlation_id",
        "selection_id",
        "downstream_consumer_id",
    )
    @classmethod
    def _validate_handoff_ids(cls, value: str, info: ValidationInfo) -> str:
        return require_non_empty_text(value, label=str(info.field_name))

    @field_validator("tenant_id")
    @classmethod
    def _validate_tenant_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_non_empty_text(value, label="tenant_id")

    @model_validator(mode="after")
    def _correlations_align(self) -> CapabilityHandoffEnvelope:
        if (
            self.discovery_correlation_id
            != self.discovery_trace.discovery_correlation_id
        ):
            raise ValueError("discovery_correlation_id must match discovery_trace")
        if self.selection_id != self.explicit_selection.selection_id:
            raise ValueError("selection_id must match explicit_selection")
        if (
            self.discovery_correlation_id
            != self.explicit_selection.discovery_correlation_id
        ):
            raise ValueError("discovery_correlation_id must match explicit_selection")
        if self.selected_release != self.explicit_selection.selected_release:
            raise ValueError("selected_release must match explicit_selection")
        tenant = self.tenant_id
        ctx_tenant = self.discovery_trace.marketplace_query_context.tenant_id
        if tenant != ctx_tenant:
            raise ValueError("tenant_id must match marketplace_query_context.tenant_id")
        if self.recorded_at.tzinfo is None:
            raise ValueError("recorded_at must be timezone-aware")
        return self


class CapabilityHandoffDeliveryDisposition(StrEnum):
    DELIVERED = "delivered"
    DUPLICATE_SKIPPED = "duplicate_skipped"
    IN_PROGRESS_SKIPPED = "in_progress_skipped"


class CapabilityHandoffDeliveryLifecycleState(StrEnum):
    """Logical delivery lifecycle — provider-scoped, not distributed exactly-once."""

    IN_PROGRESS = "in_progress"
    FAILED_RETRYABLE = "failed_retryable"
    DELIVERED = "delivered"


class CapabilityHandoffDeliveryAdmissionVerdict(StrEnum):
    """Result of atomic handoff reservation — delivery control, not observational trace."""

    RESERVED_NEW = "reserved_new"
    ALREADY_DELIVERED_IDENTICAL = "already_delivered_identical"
    IN_PROGRESS_IDENTICAL = "in_progress_identical"


class CapabilityHandoffDeliveryAdmissionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: CapabilityHandoffDeliveryAdmissionVerdict
    handoff_id: str = Field(min_length=1)


class CapabilityHandoffIdentityConflictError(Exception):
    """Same handoff_id was already admitted with a different envelope payload."""


class CapabilityHandoffDeliveryAdmissionError(Exception):
    """Delivery idempotency authority failed — delivery must not proceed."""


class CapabilityHandoffDeliveryLifecycleTransitionError(
    CapabilityHandoffDeliveryAdmissionError
):
    """Lifecycle transition failed after a bounded delivery step — must not be ignored."""


class CapabilityHandoffDeliveryOutcomeUncertainError(
    CapabilityHandoffDeliveryAdmissionError
):
    """Consumer may have succeeded but delivered state could not be committed."""


class CapabilityHandoffDeliveryLifecycleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    handoff_id: str = Field(min_length=1)
    envelope: CapabilityHandoffEnvelope
    state: CapabilityHandoffDeliveryLifecycleState


class CapabilityHandoffDeliveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    disposition: CapabilityHandoffDeliveryDisposition
    handoff_id: str = Field(min_length=1)
    downstream_consumer_id: str = Field(min_length=1)


@runtime_checkable
class CapabilityHandoffConsumer(Protocol):
    """Structural downstream handoff consumer — no platform subclass required."""

    @property
    def consumer_id(self) -> str:
        """Stable consumer identifier recorded on the envelope."""
        ...

    def consume(self, envelope: CapabilityHandoffEnvelope) -> None:
        """Accept handoff envelope; raise ``CapabilityHandoffConsumerError`` with typed disposition."""
        ...


@runtime_checkable
class CapabilityHandoffDeliveryAdmission(Protocol):
    """Mandatory delivery lifecycle authority — independent of trace persistence."""

    def reserve(
        self, envelope: CapabilityHandoffEnvelope
    ) -> CapabilityHandoffDeliveryAdmissionResult:
        """Atomically reserve a delivery attempt; raise ``CapabilityHandoffIdentityConflictError`` on clash."""
        ...

    def mark_delivered(self, handoff_id: str) -> None:
        """Commit successful delivery after downstream consumer succeeds."""
        ...

    def mark_delivery_failed(self, handoff_id: str) -> None:
        """Release an in-progress reservation so an identical envelope may retry."""
        ...


@runtime_checkable
class CapabilityHandoffTraceEvidenceConsumer(Protocol):
    """Optional observational persistence for handoff facts — not execution lineage."""

    def record_handoff(self, envelope: CapabilityHandoffEnvelope) -> bool:
        """Return True when a new canonical fact was recorded; False when deduped."""
        ...


class CapabilityHandoffConsumerFailureDisposition(StrEnum):
    """Public consumer failure semantics — not inferred from exception message text."""

    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"
    REQUIRES_HITL = "requires_hitl"
    FAILED = "failed"


class CapabilityHandoffConsumerError(Exception):
    """Typed failure from a handoff consumer — must not be swallowed."""

    disposition: CapabilityHandoffConsumerFailureDisposition
    detail: str

    def __init__(
        self,
        detail: str,
        *,
        disposition: CapabilityHandoffConsumerFailureDisposition = (
            CapabilityHandoffConsumerFailureDisposition.FAILED
        ),
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.disposition = disposition


__all__ = [
    "CapabilityDiscoveryTraceFacts",
    "CapabilityHandoffConsumer",
    "CapabilityHandoffConsumerError",
    "CapabilityHandoffConsumerFailureDisposition",
    "CapabilityHandoffConsumerTarget",
    "CapabilityHandoffDeliveryAdmission",
    "CapabilityHandoffDeliveryAdmissionError",
    "CapabilityHandoffDeliveryAdmissionResult",
    "CapabilityHandoffDeliveryAdmissionVerdict",
    "CapabilityHandoffDeliveryDisposition",
    "CapabilityHandoffDeliveryLifecycleRecord",
    "CapabilityHandoffDeliveryLifecycleState",
    "CapabilityHandoffDeliveryLifecycleTransitionError",
    "CapabilityHandoffDeliveryOutcomeUncertainError",
    "CapabilityHandoffDeliveryResult",
    "CapabilityHandoffEnvelope",
    "CapabilityHandoffIdentityConflictError",
    "CapabilityHandoffTraceEvidenceConsumer",
    "CapabilityMarketplaceExplicitSelection",
    "SCHEMA_CAPABILITY_DISCOVERY_TRACE_FACTS_V1",
    "SCHEMA_CAPABILITY_HANDOFF_ENVELOPE_V1",
    "SCHEMA_CAPABILITY_MARKETPLACE_EXPLICIT_SELECTION_V1",
    "consumer_target_for_kind",
]

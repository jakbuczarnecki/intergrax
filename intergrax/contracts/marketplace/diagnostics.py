# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pluggable marketplace pipeline diagnostics — observational only (ME-10)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.release_identity import CapabilityReleaseIdentity

SCHEMA_MARKETPLACE_DIAGNOSTIC_EVENT_V1: Final = "marketplace_diagnostic_event.v1"
SCHEMA_MARKETPLACE_OBSERVATION_CONTEXT_V1: Final = "marketplace_observation_context.v1"

NOOP_MARKETPLACE_DIAGNOSTIC_OBSERVER_ID: Final = "marketplace.diagnostics.noop"


class MarketplacePipelineStage(StrEnum):
    DISCOVERY = "discovery"
    VISIBILITY = "visibility"
    SEARCH = "search"
    RANKING = "ranking"
    GOVERNANCE = "governance"
    RECOMMENDATION = "recommendation"
    SELECTION = "selection"
    HANDOFF = "handoff"


class MarketplaceDiagnosticEventKind(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    FILTERED = "filtered"
    REJECTED = "rejected"
    FAILED = "failed"


class MarketplaceDiagnosticOutcome(StrEnum):
    SUCCESS = "success"
    EMPTY = "empty"
    REJECTED = "rejected"
    FAILED = "failed"


class MarketplaceObserverFailurePolicy(StrEnum):
    """Explicit policy when a diagnostic observer raises."""

    BEST_EFFORT = "best_effort"
    STRICT = "strict"


class MarketplaceObservationContext(BaseModel):
    """Correlation carrier — references canonical IDs only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["marketplace_observation_context.v1"] = (
        SCHEMA_MARKETPLACE_OBSERVATION_CONTEXT_V1
    )
    discovery_correlation_id: str = Field(min_length=1)
    query_correlation_id: str | None = None

    @field_validator("discovery_correlation_id")
    @classmethod
    def _validate_discovery_correlation_id(cls, value: str) -> str:
        return require_non_empty_text(value, label="discovery_correlation_id")

    def effective_query_correlation_id(self) -> str:
        return self.query_correlation_id or self.discovery_correlation_id


class MarketplaceDiagnosticEvent(BaseModel):
    """Immutable structured marketplace pipeline diagnostic fact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["marketplace_diagnostic_event.v1"] = (
        SCHEMA_MARKETPLACE_DIAGNOSTIC_EVENT_V1
    )
    stage: MarketplacePipelineStage
    event_kind: MarketplaceDiagnosticEventKind
    correlation: MarketplaceObservationContext
    outcome: MarketplaceDiagnosticOutcome = MarketplaceDiagnosticOutcome.SUCCESS
    strategy_id: str | None = None
    ranker_id: str | None = None
    recommendation_strategy_id: str | None = None
    governance_evaluator_ids: tuple[str, ...] = ()
    governance_evidence_refs: tuple[str, ...] = ()
    input_count: int | None = Field(default=None, ge=0)
    output_count: int | None = Field(default=None, ge=0)
    filtered_count: int | None = Field(default=None, ge=0)
    allowed_count: int | None = Field(default=None, ge=0)
    blocked_count: int | None = Field(default=None, ge=0)
    selected_release: CapabilityReleaseIdentity | None = None
    handoff_domain_authority_id: str | None = None
    handoff_reason_code: str | None = None
    handoff_status: str | None = None
    duration_ms: float | None = Field(default=None, ge=0.0)
    detail: str | None = None


@runtime_checkable
class MarketplaceDiagnosticObserver(Protocol):
    """Structural diagnostic sink — must not influence business decisions."""

    @property
    def observer_id(self) -> str:
        """Stable observer identifier (not a class name)."""

    def emit(self, event: MarketplaceDiagnosticEvent) -> None:
        """Receive one diagnostic event."""


class MarketplaceObserverEmitError(RuntimeError):
    """Raised when observer emission fails under STRICT policy."""


__all__ = [
    "NOOP_MARKETPLACE_DIAGNOSTIC_OBSERVER_ID",
    "MarketplaceDiagnosticEvent",
    "MarketplaceDiagnosticEventKind",
    "MarketplaceDiagnosticObserver",
    "MarketplaceDiagnosticOutcome",
    "MarketplaceObservationContext",
    "MarketplaceObserverEmitError",
    "MarketplaceObserverFailurePolicy",
    "MarketplacePipelineStage",
    "SCHEMA_MARKETPLACE_DIAGNOSTIC_EVENT_V1",
    "SCHEMA_MARKETPLACE_OBSERVATION_CONTEXT_V1",
]

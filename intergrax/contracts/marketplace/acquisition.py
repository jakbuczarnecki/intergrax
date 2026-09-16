# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Machine capability acquisition contracts (ME-12)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.need import CapabilityNeed
from intergrax.contracts.capability_catalog.query import CapabilityDiscoveryQuery
from intergrax.contracts.capability_catalog.release_identity import CapabilityReleaseIdentity
from intergrax.contracts.capability_catalog.recommendation import (
    CapabilityRecommendationContext,
    CapabilityRecommendationEvidence,
)
from intergrax.contracts.marketplace.diagnostics import MarketplaceObservationContext
from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffConsumerTarget,
    CapabilityHandoffDeliveryResult,
)
from intergrax.contracts.marketplace.query_context import MarketplaceQueryContext

SCHEMA_MACHINE_CAPABILITY_ACQUISITION_REQUEST_V1: Final = (
    "machine_capability_acquisition_request.v1"
)
SCHEMA_MACHINE_CAPABILITY_ACQUISITION_RESPONSE_V1: Final = (
    "machine_capability_acquisition_response.v1"
)
SCHEMA_MACHINE_CAPABILITY_ACQUISITION_SELECTION_V1: Final = (
    "machine_capability_acquisition_selection.v1"
)
SCHEMA_MACHINE_CAPABILITY_RECOMMENDATION_V1: Final = "machine_capability_recommendation.v1"
SCHEMA_MACHINE_CAPABILITY_ACQUISITION_HANDOFF_RESPONSE_V1: Final = (
    "machine_capability_acquisition_handoff_response.v1"
)


class MachineCapabilityRecommendation(BaseModel):
    """Public machine projection of a governed advisory recommendation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["machine_capability_recommendation.v1"] = (
        SCHEMA_MACHINE_CAPABILITY_RECOMMENDATION_V1
    )
    release: CapabilityReleaseIdentity
    recommendation_evidence: CapabilityRecommendationEvidence
    governance_evidence_refs: tuple[str, ...] = ()
    ranking_strategy_id: str | None = None


class MachineCapabilityAcquisitionOutcome(StrEnum):
    RECOMMENDATIONS_AVAILABLE = "recommendations_available"
    NO_MATCH = "no_match"
    NO_GOVERNED_MATCH = "no_governed_match"
    NO_RECOMMENDATIONS = "no_recommendations"


class MachineCatalogFederationCompleteness(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"


class MachineCapabilityAcquisitionRequest(BaseModel):
    """Typed acquisition request — discovery query and need stay distinct."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["machine_capability_acquisition_request.v1"] = (
        SCHEMA_MACHINE_CAPABILITY_ACQUISITION_REQUEST_V1
    )
    request_id: str = Field(min_length=1)
    need: CapabilityNeed
    discovery_query: CapabilityDiscoveryQuery
    marketplace_query_context: MarketplaceQueryContext = Field(
        default_factory=MarketplaceQueryContext,
    )
    observation: MarketplaceObservationContext | None = None
    recommendation_context: CapabilityRecommendationContext | None = None
    query_text: str | None = None

    @field_validator("request_id")
    @classmethod
    def _validate_request_id(cls, value: str) -> str:
        return require_non_empty_text(value, label="request_id")


class MachineCapabilityAcquisitionResponse(BaseModel):
    """Governed recommendations only — never raw ranked or blocked candidates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["machine_capability_acquisition_response.v1"] = (
        SCHEMA_MACHINE_CAPABILITY_ACQUISITION_RESPONSE_V1
    )
    request_id: str = Field(min_length=1)
    discovery_correlation_id: str = Field(min_length=1)
    outcome: MachineCapabilityAcquisitionOutcome
    recommendations: tuple[MachineCapabilityRecommendation, ...] = ()
    observation: MarketplaceObservationContext | None = None
    catalog_federation_completeness: MachineCatalogFederationCompleteness | None = None

    @field_validator("request_id", "discovery_correlation_id")
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return require_non_empty_text(value, label=str(info.field_name))

    @model_validator(mode="after")
    def _outcome_matches_recommendations(self) -> MachineCapabilityAcquisitionResponse:
        if self.outcome is MachineCapabilityAcquisitionOutcome.RECOMMENDATIONS_AVAILABLE:
            if not self.recommendations:
                raise ValueError(
                    "RECOMMENDATIONS_AVAILABLE requires non-empty recommendations",
                )
        elif self.recommendations:
            raise ValueError("empty outcome must not carry recommendations")
        return self


class MachineCapabilityAcquisitionSelection(BaseModel):
    """Explicit machine selection — must reference an exact governed release."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["machine_capability_acquisition_selection.v1"] = (
        SCHEMA_MACHINE_CAPABILITY_ACQUISITION_SELECTION_V1
    )
    selection_id: str = Field(min_length=1)
    discovery_correlation_id: str = Field(min_length=1)
    selected_release: CapabilityReleaseIdentity
    selector_id: str = Field(min_length=1)
    consumer_target: CapabilityHandoffConsumerTarget | None = None

    @field_validator("selection_id", "discovery_correlation_id", "selector_id")
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return require_non_empty_text(value, label=str(info.field_name))


class MachineCapabilityAcquisitionHandoffRequest(BaseModel):
    """Re-runs the marketplace pipeline to validate selection against recommendations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["machine_capability_acquisition_request.v1"] = (
        SCHEMA_MACHINE_CAPABILITY_ACQUISITION_REQUEST_V1
    )
    acquisition_request: MachineCapabilityAcquisitionRequest
    selection: MachineCapabilityAcquisitionSelection
    handoff_id: str = Field(min_length=1)

    @field_validator("handoff_id")
    @classmethod
    def _validate_handoff_id(cls, value: str) -> str:
        return require_non_empty_text(value, label="handoff_id")

    @model_validator(mode="after")
    def _correlation_aligns(self) -> MachineCapabilityAcquisitionHandoffRequest:
        observation = self.acquisition_request.observation
        if observation is not None:
            if self.selection.discovery_correlation_id != observation.discovery_correlation_id:
                raise ValueError(
                    "selection.discovery_correlation_id must match acquisition observation",
                )
        return self


class MachineCapabilityAcquisitionHandoffResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["machine_capability_acquisition_handoff_response.v1"] = (
        SCHEMA_MACHINE_CAPABILITY_ACQUISITION_HANDOFF_RESPONSE_V1
    )
    request_id: str = Field(min_length=1)
    discovery_correlation_id: str = Field(min_length=1)
    selection_id: str = Field(min_length=1)
    handoff_id: str = Field(min_length=1)
    delivery: CapabilityHandoffDeliveryResult

    @field_validator("request_id", "discovery_correlation_id", "selection_id", "handoff_id")
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return require_non_empty_text(value, label=str(info.field_name))


@runtime_checkable
class MachineCapabilityAcquisitionPolicy(Protocol):
    """May only narrow governed recommendations — never widen or bypass governance."""

    def narrow_recommendations(
        self,
        recommendations: tuple[MachineCapabilityRecommendation, ...],
    ) -> tuple[MachineCapabilityRecommendation, ...]:
        """Return a subset of the input recommendations."""
        ...


__all__ = [
    "MachineCapabilityAcquisitionHandoffRequest",
    "MachineCapabilityAcquisitionHandoffResponse",
    "MachineCapabilityAcquisitionOutcome",
    "MachineCapabilityRecommendation",
    "SCHEMA_MACHINE_CAPABILITY_RECOMMENDATION_V1",
    "MachineCatalogFederationCompleteness",
    "MachineCapabilityAcquisitionPolicy",
    "MachineCapabilityAcquisitionRequest",
    "MachineCapabilityAcquisitionResponse",
    "MachineCapabilityAcquisitionSelection",
    "SCHEMA_MACHINE_CAPABILITY_ACQUISITION_HANDOFF_RESPONSE_V1",
    "SCHEMA_MACHINE_CAPABILITY_ACQUISITION_REQUEST_V1",
    "SCHEMA_MACHINE_CAPABILITY_ACQUISITION_RESPONSE_V1",
    "SCHEMA_MACHINE_CAPABILITY_ACQUISITION_SELECTION_V1",
]

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Gap-anchored marketplace acquisition — listing/source resolution after canonical Gap (UCA-5)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.need import CapabilityNeed
from intergrax.contracts.capability_catalog.query import CapabilityDiscoveryQuery
from intergrax.contracts.marketplace.query_context import MarketplaceQueryContext

SCHEMA_MARKETPLACE_GAP_ACQUISITION_REQUEST_V1: Final = (
    "marketplace_gap_acquisition_request.v1"
)
SCHEMA_MARKETPLACE_GAP_ACQUISITION_RESULT_V1: Final = (
    "marketplace_gap_acquisition_result.v1"
)


class MarketplaceGapAcquisitionOutcome(StrEnum):
    """Marketplace domain outcome — not UCA qualification or execution."""

    SUCCEEDED = "succeeded"
    NO_ACQUISITION_SOURCE = "no_acquisition_source"
    UNAVAILABLE = "unavailable"
    BLOCKED = "blocked"
    REQUIRES_HITL = "requires_hitl"
    FAILED = "failed"


class MarketplaceGapAcquisitionRequest(BaseModel):
    """Resolve marketplace offers and perform governed handoff for an existing CapabilityGap.

    Does not re-run canonical platform capability discovery (UCA-1). Listing visibility
    search is marketplace acquisition-source resolution only.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["marketplace_gap_acquisition_request.v1"] = (
        SCHEMA_MARKETPLACE_GAP_ACQUISITION_REQUEST_V1
    )
    operation_id: str = Field(min_length=1)
    gap_id: str = Field(min_length=1)
    canonical_discovery_correlation_id: str = Field(min_length=1)
    capability_need: CapabilityNeed
    discovery_query: CapabilityDiscoveryQuery
    marketplace_query_context: MarketplaceQueryContext = Field(
        default_factory=MarketplaceQueryContext,
    )
    query_text: str | None = None
    correlation_id: str | None = None
    causation_id: str | None = None
    selector_id: str = Field(min_length=1, default="uca.marketplace.gap_acquisition")

    @field_validator(
        "operation_id",
        "gap_id",
        "canonical_discovery_correlation_id",
        "selector_id",
        "correlation_id",
        "causation_id",
    )
    @classmethod
    def _validate_ids(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_non_empty_text(value, label=str(info.field_name))


class MarketplaceGapAcquisitionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["marketplace_gap_acquisition_result.v1"] = (
        SCHEMA_MARKETPLACE_GAP_ACQUISITION_RESULT_V1
    )
    operation_id: str = Field(min_length=1)
    gap_id: str = Field(min_length=1)
    outcome: MarketplaceGapAcquisitionOutcome
    marketplace_listing_correlation_id: str | None = None
    domain_handoff_reference: str | None = None
    artifact_reference: str | None = None
    reason_detail: str = ""

    @field_validator("operation_id", "gap_id")
    @classmethod
    def _validate_required_ids(cls, value: str, info) -> str:
        return require_non_empty_text(value, label=str(info.field_name))


@runtime_checkable
class MarketplaceGapAcquisitionPort(Protocol):
    """Public marketplace seam for UCA gap acquisition — replaceable implementation."""

    def acquire_from_gap(
        self,
        request: MarketplaceGapAcquisitionRequest,
    ) -> MarketplaceGapAcquisitionResult:
        """Resolve acquisition source and complete governed domain handoff when possible."""
        ...


__all__ = [
    "MarketplaceGapAcquisitionOutcome",
    "MarketplaceGapAcquisitionPort",
    "MarketplaceGapAcquisitionRequest",
    "MarketplaceGapAcquisitionResult",
    "SCHEMA_MARKETPLACE_GAP_ACQUISITION_REQUEST_V1",
    "SCHEMA_MARKETPLACE_GAP_ACQUISITION_RESULT_V1",
]

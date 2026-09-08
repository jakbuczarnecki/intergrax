# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace capability listing product contract (Stage 11)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.contracts.capability_catalog._validation import normalize_optional_text
from intergrax.contracts.capability_catalog.availability import AvailabilityDisposition
from intergrax.contracts.marketplace.commercial import MarketplaceCommercialMetadata
from intergrax.contracts.marketplace.publisher import MarketplacePublisherMetadata

SCHEMA_MARKETPLACE_CAPABILITY_LISTING_V1: Final = "marketplace_capability_listing.v1"
SCHEMA_MARKETPLACE_CAPABILITY_LISTING_VIEW_V1: Final = (
    "marketplace_capability_listing_view.v1"
)


class MarketplaceCapabilityListing(BaseModel):
    """Product surface wrapping a canonical catalog entry — metadata never rewrites entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["marketplace_capability_listing.v1"] = (
        SCHEMA_MARKETPLACE_CAPABILITY_LISTING_V1
    )
    listing_id: str | None = None
    capability: CapabilityCatalogEntry
    publisher_metadata: MarketplacePublisherMetadata | None = None
    commercial_metadata: MarketplaceCommercialMetadata | None = None

    @field_validator("listing_id")
    @classmethod
    def _validate_listing_id(cls, value: str | None) -> str | None:
        return normalize_optional_text(value, label="listing_id")

    @model_validator(mode="after")
    def _validate_publisher_consistency(self) -> MarketplaceCapabilityListing:
        provenance_publisher = self.capability.provenance.publisher
        if self.publisher_metadata is None or provenance_publisher is None:
            return self
        if self.publisher_metadata.publisher_id != provenance_publisher:
            raise ValueError(
                "publisher_metadata.publisher_id must equal capability.provenance.publisher",
            )
        return self


class MarketplaceCapabilityListingView(BaseModel):
    """Read-only listing projection joined with discovery availability evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["marketplace_capability_listing_view.v1"] = (
        SCHEMA_MARKETPLACE_CAPABILITY_LISTING_VIEW_V1
    )
    listing: MarketplaceCapabilityListing
    availability: AvailabilityDisposition = Field(
        description="Projected availability — CATALOG_AVAILABLE is not HOST_AVAILABLE.",
    )

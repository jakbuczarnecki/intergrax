# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace product contracts (CAPABILITY-CATALOG-1 Stage 11)."""

from __future__ import annotations

from intergrax.contracts.marketplace.commercial import (
    SCHEMA_MARKETPLACE_COMMERCIAL_METADATA_V1,
    CommercialModel,
    MarketplaceCommercialMetadata,
)
from intergrax.contracts.marketplace.listing import (
    SCHEMA_MARKETPLACE_CAPABILITY_LISTING_V1,
    SCHEMA_MARKETPLACE_CAPABILITY_LISTING_VIEW_V1,
    MarketplaceCapabilityListing,
    MarketplaceCapabilityListingView,
)
from intergrax.contracts.marketplace.listing_projection import MarketplaceListingProjection
from intergrax.contracts.marketplace.listing_record import MarketplaceListingRecord
from intergrax.contracts.marketplace.metadata_source import MarketplaceMetadataSource
from intergrax.contracts.marketplace.publisher import (
    SCHEMA_MARKETPLACE_PUBLISHER_METADATA_V1,
    MarketplacePublisherMetadata,
)

__all__ = [
    "CommercialModel",
    "MarketplaceCapabilityListing",
    "MarketplaceCapabilityListingView",
    "MarketplaceCommercialMetadata",
    "MarketplaceListingProjection",
    "MarketplaceListingRecord",
    "MarketplaceMetadataSource",
    "MarketplacePublisherMetadata",
    "SCHEMA_MARKETPLACE_CAPABILITY_LISTING_V1",
    "SCHEMA_MARKETPLACE_CAPABILITY_LISTING_VIEW_V1",
    "SCHEMA_MARKETPLACE_COMMERCIAL_METADATA_V1",
    "SCHEMA_MARKETPLACE_PUBLISHER_METADATA_V1",
]

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Read-only marketplace product surface (CAPABILITY-CATALOG-1 Stage 11)."""

from __future__ import annotations

from intergrax.marketplace.errors import (
    MarketplaceCatalogConfigurationError,
    MarketplaceCatalogError,
)
from intergrax.marketplace.listing import (
    SCHEMA_MARKETPLACE_CAPABILITY_LISTING_V1,
    SCHEMA_MARKETPLACE_CAPABILITY_LISTING_VIEW_V1,
    MarketplaceCapabilityListing,
    MarketplaceCapabilityListingView,
)
from intergrax.marketplace.projection import (
    DefaultMarketplaceListingProjection,
    build_marketplace_listing,
    default_marketplace_listing_projection,
    project_marketplace_record,
)
from intergrax.marketplace.record import MarketplaceListingRecord
from intergrax.marketplace.discovery import MarketplaceDiscoveryService
from intergrax.marketplace.search import (
    MARKETPLACE_LISTING_TEXT_SEARCH_STRATEGY_ID,
    DefaultMarketplaceListingTextSearchStrategy,
)
from intergrax.marketplace.service import MarketplaceCatalogService, snapshot_without_marketplace
from intergrax.marketplace.snapshot import (
    SCHEMA_MARKETPLACE_CATALOG_SNAPSHOT_V1,
    MarketplaceCatalogSnapshot,
)
from intergrax.marketplace.source import (
    InMemoryMarketplaceMetadataSource,
    MarketplaceCapabilityCatalogSource,
)

__all__ = [
    "DefaultMarketplaceListingProjection",
    "InMemoryMarketplaceMetadataSource",
    "MarketplaceCapabilityCatalogSource",
    "MarketplaceCapabilityListing",
    "MarketplaceCapabilityListingView",
    "MarketplaceCatalogConfigurationError",
    "MarketplaceCatalogError",
    "DefaultMarketplaceListingTextSearchStrategy",
    "MARKETPLACE_LISTING_TEXT_SEARCH_STRATEGY_ID",
    "MarketplaceCatalogService",
    "MarketplaceDiscoveryService",
    "MarketplaceCatalogSnapshot",
    "MarketplaceListingRecord",
    "SCHEMA_MARKETPLACE_CAPABILITY_LISTING_V1",
    "SCHEMA_MARKETPLACE_CAPABILITY_LISTING_VIEW_V1",
    "SCHEMA_MARKETPLACE_CATALOG_SNAPSHOT_V1",
    "build_marketplace_listing",
    "default_marketplace_listing_projection",
    "project_marketplace_record",
    "snapshot_without_marketplace",
]

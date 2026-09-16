# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Read-only marketplace capability catalog source adapter (Stage 11)."""

from __future__ import annotations

from intergrax.contracts.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.contracts.capability_catalog.identity import CapabilitySourceIdentity
from intergrax.contracts.marketplace.listing import MarketplaceCapabilityListing
from intergrax.contracts.marketplace.listing_projection import MarketplaceListingProjection
from intergrax.contracts.marketplace.listing_record import MarketplaceListingRecord
from intergrax.marketplace.projection import DefaultMarketplaceListingProjection
from intergrax.marketplace.snapshot import MarketplaceCatalogSnapshot
from intergrax.marketplace.validation import validate_marketplace_source


class MarketplaceCapabilityCatalogSource:
    """Default in-memory marketplace metadata backend (also federates via ``read_entries``)."""

    def __init__(
        self,
        *,
        source: CapabilitySourceIdentity,
        records: tuple[MarketplaceListingRecord, ...],
        listing_projection: MarketplaceListingProjection | None = None,
    ) -> None:
        validate_marketplace_source(source)
        projection = listing_projection or DefaultMarketplaceListingProjection()
        listings = tuple(projection.build_listing(source, record) for record in records)
        self._snapshot = MarketplaceCatalogSnapshot(source=source, listings=listings)

    @property
    def source_id(self) -> str:
        return self._snapshot.source.source_id

    @property
    def source(self) -> CapabilitySourceIdentity:
        return self._snapshot.source

    def read_listings(self) -> tuple[MarketplaceCapabilityListing, ...]:
        return self._snapshot.listings

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        entries = [listing.capability for listing in self._snapshot.listings]
        return tuple(sorted(entries, key=lambda entry: entry.identity.sort_key))


InMemoryMarketplaceMetadataSource = MarketplaceCapabilityCatalogSource

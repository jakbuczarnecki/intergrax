# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Read-only marketplace product query surface (Stage 11)."""

from __future__ import annotations

from intergrax.capability_catalog.discovery import discover_capability_candidates
from intergrax.capability_catalog.federation import FederatedCapabilityCatalog
from intergrax.capability_catalog.snapshot import CapabilityCatalogSnapshot
from intergrax.contracts.capability_catalog.evidence import (
    CapabilityDiscoveryAvailabilityEvidence,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.query import CapabilityDiscoveryQuery
from intergrax.marketplace.listing import (
    MarketplaceCapabilityListing,
    MarketplaceCapabilityListingView,
)
from intergrax.marketplace.source import MarketplaceCapabilityCatalogSource


class MarketplaceCatalogService:
    """Join federated catalog discovery with marketplace product metadata."""

    def __init__(
        self,
        *,
        catalog: FederatedCapabilityCatalog,
        marketplace_sources: tuple[MarketplaceCapabilityCatalogSource, ...],
    ) -> None:
        self._catalog = catalog
        self._listing_index = _build_listing_index(marketplace_sources)

    def list_listings(
        self,
        query: CapabilityDiscoveryQuery,
        *,
        availability_evidence: CapabilityDiscoveryAvailabilityEvidence | None = None,
        query_text: str | None = None,
    ) -> tuple[MarketplaceCapabilityListingView, ...]:
        """List marketplace listings matching a Stage-3 discovery query."""
        snapshot = self._catalog.snapshot()
        candidates = discover_capability_candidates(
            snapshot,
            query,
            availability_evidence=availability_evidence,
        )
        normalized_query = _normalize_query_text(query_text)
        views: list[MarketplaceCapabilityListingView] = []
        for candidate in candidates:
            listing = self._listing_index.get(candidate.identity.sort_key)
            if listing is None:
                continue
            if normalized_query is not None and not _matches_query_text(
                listing,
                normalized_query,
            ):
                continue
            views.append(
                MarketplaceCapabilityListingView(
                    listing=listing,
                    availability=candidate.availability,
                ),
            )
        return tuple(views)

    def get_listing(
        self,
        identity_key: CapabilityIdentityKey,
    ) -> MarketplaceCapabilityListing | None:
        """Return one marketplace listing by canonical identity key, if present."""
        return self._listing_index.get(identity_key.sort_key)


def _build_listing_index(
    marketplace_sources: tuple[MarketplaceCapabilityCatalogSource, ...],
) -> dict[tuple[str, str, str, str], MarketplaceCapabilityListing]:
    index: dict[tuple[str, str, str, str], MarketplaceCapabilityListing] = {}
    for source in marketplace_sources:
        if source.source_id in {listing.capability.identity.source.source_id for listing in index.values()}:
            continue
        for listing in source.read_listings():
            key = listing.capability.identity.sort_key
            index[key] = listing
    return index


def _normalize_query_text(query_text: str | None) -> str | None:
    if query_text is None:
        return None
    normalized = query_text.strip().casefold()
    if not normalized:
        return None
    return normalized


def _matches_query_text(
    listing: MarketplaceCapabilityListing,
    query_text: str,
) -> bool:
    capability = listing.capability
    haystacks = (
        capability.identity.logical.logical_id.casefold(),
        (capability.display_label or "").casefold(),
    )
    return any(query_text in haystack for haystack in haystacks)


def snapshot_without_marketplace(
    catalog: FederatedCapabilityCatalog,
) -> CapabilityCatalogSnapshot:
    """Expose federated snapshot for air-gapped callers — no marketplace dependency."""
    return catalog.snapshot()

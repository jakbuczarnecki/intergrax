# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Read-only marketplace product query surface (Stage 11)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.capability_catalog.candidate import CapabilityDiscoveryCandidate
from intergrax.capability_catalog.discovery import discover_capability_candidates
from intergrax.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.capability_catalog.search import search_capability_candidates
from intergrax.capability_catalog.snapshot_provider import CapabilityCatalogSnapshotProvider
from intergrax.capability_catalog.snapshot import CapabilityCatalogSnapshot
from intergrax.contracts.capability_catalog.evidence import (
    CapabilityDiscoveryAvailabilityEvidence,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.query import CapabilityDiscoveryQuery
from intergrax.contracts.capability_catalog.search import (
    CapabilitySearchContext,
    CapabilitySearchQuery,
)
from intergrax.contracts.marketplace import (
    MarketplaceCommercialMetadata,
    MarketplacePublisherMetadata,
)
from intergrax.marketplace.errors import MarketplaceCatalogConfigurationError
from intergrax.contracts.marketplace.metadata_source import MarketplaceMetadataSource
from intergrax.capability_catalog.search import CapabilitySearchStrategy
from intergrax.marketplace.listing import (
    MarketplaceCapabilityListing,
    MarketplaceCapabilityListingView,
)
from intergrax.contracts.marketplace.query_context import MarketplaceQueryContext
from intergrax.contracts.marketplace.visibility import MarketplaceVisibility
from intergrax.marketplace.search import DefaultMarketplaceListingTextSearchStrategy
from intergrax.marketplace.visibility import MarketplaceVisibilityEvaluator
from intergrax.contracts.marketplace.diagnostics import (
    MarketplaceDiagnosticEvent,
    MarketplaceDiagnosticEventKind,
    MarketplaceDiagnosticOutcome,
    MarketplacePipelineStage,
)
from intergrax.marketplace.diagnostics import emit_marketplace_diagnostic
from intergrax.marketplace.diagnostics.session import MarketplacePipelineObservationSession
from intergrax.capability_catalog.snapshot import CapabilityCatalogFederationCompleteness


@dataclass(frozen=True, slots=True)
class MarketplaceCatalogListingQueryResult:
    """Public listing query view — listings and federation facts from one catalog read."""

    listing_views: tuple[MarketplaceCapabilityListingView, ...]
    catalog_federation_completeness: CapabilityCatalogFederationCompleteness


@dataclass(frozen=True, slots=True)
class _ListingProductMetadata:
    listing_id: str | None
    publisher_metadata: MarketplacePublisherMetadata | None
    commercial_metadata: MarketplaceCommercialMetadata | None
    visibility: MarketplaceVisibility | None


class MarketplaceCatalogService:
    """Join federated catalog discovery with marketplace product metadata.

    Construction performs static federation/wiring validation only. Each
    ``query_listings`` / ``get_listing`` call reads the current metadata snapshot
    from every configured source, validates it, then joins against one catalog
    snapshot for that query.
    """

    def __init__(
        self,
        *,
        catalog: CapabilityCatalogSnapshotProvider,
        marketplace_sources: tuple[MarketplaceMetadataSource, ...],
        listing_text_search: CapabilitySearchStrategy | None = None,
        visibility_evaluator: MarketplaceVisibilityEvaluator | None = None,
    ) -> None:
        self._catalog = catalog
        self._listing_text_search = (
            listing_text_search or DefaultMarketplaceListingTextSearchStrategy()
        )
        self._visibility_evaluator = (
            visibility_evaluator or MarketplaceVisibilityEvaluator()
        )
        self._marketplace_sources = marketplace_sources
        _validate_marketplace_sources_in_catalog(catalog, marketplace_sources)
        _validate_marketplace_metadata_source_configuration(marketplace_sources)

    def list_listings(
        self,
        query: CapabilityDiscoveryQuery,
        *,
        marketplace_query_context: MarketplaceQueryContext | None = None,
        availability_evidence: CapabilityDiscoveryAvailabilityEvidence | None = None,
        query_text: str | None = None,
        observation: MarketplacePipelineObservationSession | None = None,
    ) -> tuple[MarketplaceCapabilityListingView, ...]:
        """List marketplace listings matching a Stage-3 discovery query."""
        return self.query_listings(
            query,
            marketplace_query_context=marketplace_query_context,
            availability_evidence=availability_evidence,
            query_text=query_text,
            observation=observation,
        ).listing_views

    def query_listings(
        self,
        query: CapabilityDiscoveryQuery,
        *,
        marketplace_query_context: MarketplaceQueryContext | None = None,
        availability_evidence: CapabilityDiscoveryAvailabilityEvidence | None = None,
        query_text: str | None = None,
        observation: MarketplacePipelineObservationSession | None = None,
    ) -> MarketplaceCatalogListingQueryResult:
        """List marketplace listings from a single federated catalog snapshot read."""
        query_context = marketplace_query_context or MarketplaceQueryContext()
        snapshot = self._catalog.snapshot()
        catalog_federation_completeness = snapshot.federation_completeness
        listing_index = _build_validated_listing_index(snapshot, self._marketplace_sources)
        candidates = discover_capability_candidates(
            snapshot,
            query,
            availability_evidence=availability_evidence,
        )
        joined_count = 0
        visibility_filtered = 0
        listing_candidates: list[CapabilityDiscoveryCandidate] = []
        for candidate in candidates:
            metadata = listing_index.get(candidate.identity.sort_key)
            if metadata is None:
                continue
            joined_count += 1
            if not self._visibility_evaluator.is_visible(
                metadata.visibility,
                query_context,
            ):
                visibility_filtered += 1
                continue
            listing_candidates.append(candidate)

        if observation is not None:
            emit_marketplace_diagnostic(
                observation,
                MarketplaceDiagnosticEvent(
                    stage=MarketplacePipelineStage.VISIBILITY,
                    event_kind=MarketplaceDiagnosticEventKind.COMPLETED,
                    correlation=observation.correlation,
                    input_count=joined_count,
                    output_count=len(listing_candidates),
                    filtered_count=visibility_filtered,
                    detail="aggregate visibility narrowing only",
                ),
            )

        pre_search_count = len(listing_candidates)
        searched = search_capability_candidates(
            tuple(listing_candidates),
            self._listing_text_search,
            query=CapabilitySearchQuery(text=query_text),
            context=CapabilitySearchContext(),
        )
        if observation is not None:
            emit_marketplace_diagnostic(
                observation,
                MarketplaceDiagnosticEvent(
                    stage=MarketplacePipelineStage.SEARCH,
                    event_kind=MarketplaceDiagnosticEventKind.COMPLETED,
                    correlation=observation.correlation,
                    strategy_id=self._listing_text_search.search_strategy_id,
                    input_count=pre_search_count,
                    output_count=len(searched),
                    outcome=(
                        MarketplaceDiagnosticOutcome.EMPTY
                        if not searched
                        else MarketplaceDiagnosticOutcome.SUCCESS
                    ),
                ),
            )
        views: list[MarketplaceCapabilityListingView] = []
        for item in searched:
            metadata = listing_index[item.candidate.identity.sort_key]
            if metadata is None:
                continue
            listing = _build_listing(item.candidate.catalog_entry, metadata)
            views.append(
                MarketplaceCapabilityListingView(
                    listing=listing,
                    availability=item.candidate.availability,
                ),
            )
        return MarketplaceCatalogListingQueryResult(
            listing_views=tuple(views),
            catalog_federation_completeness=catalog_federation_completeness,
        )

    def get_listing(
        self,
        identity_key: CapabilityIdentityKey,
        *,
        marketplace_query_context: MarketplaceQueryContext | None = None,
    ) -> MarketplaceCapabilityListing | None:
        """Return one marketplace listing by canonical identity key, if present."""
        query_context = marketplace_query_context or MarketplaceQueryContext()
        snapshot = self._catalog.snapshot()
        listing_index = _build_validated_listing_index(snapshot, self._marketplace_sources)
        metadata = listing_index.get(identity_key.sort_key)
        if metadata is None:
            return None
        if not self._visibility_evaluator.is_visible(metadata.visibility, query_context):
            return None
        canonical = _index_canonical_entries(snapshot).get(identity_key.sort_key)
        if canonical is None:
            return None
        return _build_listing(canonical, metadata)


def _index_canonical_entries(
    snapshot: CapabilityCatalogSnapshot,
) -> dict[tuple[str, str, str, str], CapabilityCatalogEntry]:
    return {entry.identity.sort_key: entry for entry in snapshot.entries}


def _validate_marketplace_sources_in_catalog(
    catalog: CapabilityCatalogSnapshotProvider,
    marketplace_sources: tuple[MarketplaceMetadataSource, ...],
) -> None:
    catalog_source_ids = {source.source_id for source in catalog.sources}
    for source in marketplace_sources:
        if source.source_id not in catalog_source_ids:
            raise MarketplaceCatalogConfigurationError(
                "marketplace catalog source must be present in federated catalog: "
                f"{source.source_id!r}",
            )


def _validate_marketplace_metadata_source_configuration(
    marketplace_sources: tuple[MarketplaceMetadataSource, ...],
) -> None:
    """Static wiring checks only — must not call ``MarketplaceMetadataSource.read_listings``."""
    seen_source_ids: set[str] = set()
    for source in marketplace_sources:
        if source.source_id in seen_source_ids:
            raise MarketplaceCatalogConfigurationError(
                f"duplicate catalog source_id in federation: {source.source_id!r}",
            )
        seen_source_ids.add(source.source_id)


def _build_validated_listing_index(
    snapshot: CapabilityCatalogSnapshot,
    marketplace_sources: tuple[MarketplaceMetadataSource, ...],
) -> dict[tuple[str, str, str, str], _ListingProductMetadata]:
    """Read each metadata source once and validate the returned snapshot before join."""
    canonical_by_identity = _index_canonical_entries(snapshot)
    index: dict[tuple[str, str, str, str], _ListingProductMetadata] = {}
    for source in marketplace_sources:
        source_listings = source.read_listings()
        for listing in source_listings:
            identity_key = listing.capability.identity.sort_key
            if identity_key in index:
                raise MarketplaceCatalogConfigurationError(
                    "duplicate marketplace listing for the same source-qualified discovery identity",
                )
            canonical = canonical_by_identity.get(identity_key)
            if canonical is not None and listing.capability != canonical:
                raise MarketplaceCatalogConfigurationError(
                    "marketplace listing canonical facts must equal federated catalog entry",
                )
            index[identity_key] = _ListingProductMetadata(
                listing_id=listing.listing_id,
                publisher_metadata=listing.publisher_metadata,
                commercial_metadata=listing.commercial_metadata,
                visibility=listing.visibility,
            )
    return index


def _build_listing(
    capability: CapabilityCatalogEntry,
    metadata: _ListingProductMetadata,
) -> MarketplaceCapabilityListing:
    return MarketplaceCapabilityListing(
        listing_id=metadata.listing_id,
        capability=capability,
        publisher_metadata=metadata.publisher_metadata,
        commercial_metadata=metadata.commercial_metadata,
        visibility=metadata.visibility,
    )


def snapshot_without_marketplace(
    catalog: CapabilityCatalogSnapshotProvider,
) -> CapabilityCatalogSnapshot:
    """Expose federated snapshot for air-gapped callers — no marketplace dependency."""
    return catalog.snapshot()

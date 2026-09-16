# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Projection helpers for marketplace catalog entries (Stage 11)."""

from __future__ import annotations

from typing import Final

from intergrax.contracts.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.contracts.capability_catalog.identity import (
    CapabilityDiscoveryIdentity,
    CapabilityLogicalIdentity,
    CapabilitySourceIdentity,
)
from intergrax.contracts.capability_catalog.provenance import CapabilityProvenance
from intergrax.contracts.marketplace.listing import MarketplaceCapabilityListing
from intergrax.contracts.marketplace.listing_projection import MarketplaceListingProjection
from intergrax.contracts.marketplace.listing_record import MarketplaceListingRecord
from intergrax.marketplace.errors import MarketplaceCatalogConfigurationError
from intergrax.marketplace.validation import (
    validate_listing_source_consistency,
    validate_marketplace_source,
)

DEFAULT_MARKETPLACE_LISTING_PROJECTION_ID: Final = "marketplace.listing_projection.default.v1"


def _resolve_publisher(record: MarketplaceListingRecord) -> str | None:
    if record.publisher_metadata is not None and record.publisher is not None:
        if record.publisher_metadata.publisher_id != record.publisher:
            raise MarketplaceCatalogConfigurationError(
                "record.publisher must equal publisher_metadata.publisher_id when both provided",
            )
    if record.publisher_metadata is not None:
        return record.publisher_metadata.publisher_id
    return record.publisher


class DefaultMarketplaceListingProjection:
    """Platform default listing projection — preserves canonical catalog semantics."""

    @property
    def projection_id(self) -> str:
        return DEFAULT_MARKETPLACE_LISTING_PROJECTION_ID

    def project_catalog_entry(
        self,
        source: CapabilitySourceIdentity,
        record: MarketplaceListingRecord,
    ) -> CapabilityCatalogEntry:
        validate_marketplace_source(source)
        logical_id = record.logical_id.strip()
        if not logical_id:
            raise MarketplaceCatalogConfigurationError("marketplace logical_id must be non-empty")
        publisher = _resolve_publisher(record)
        return CapabilityCatalogEntry(
            identity=CapabilityDiscoveryIdentity(
                kind=record.kind,
                source=source,
                logical=CapabilityLogicalIdentity(
                    kind=record.kind,
                    logical_id=logical_id,
                ),
            ),
            provenance=CapabilityProvenance(
                source=source,
                version_label=record.version_label,
                package_reference=record.package_reference,
                content_digest=record.content_digest,
                publisher=publisher,
            ),
            display_label=record.display_label or logical_id,
        )

    def build_listing(
        self,
        source: CapabilitySourceIdentity,
        record: MarketplaceListingRecord,
    ) -> MarketplaceCapabilityListing:
        capability = self.project_catalog_entry(source, record)
        listing = MarketplaceCapabilityListing(
            listing_id=record.listing_id,
            capability=capability,
            publisher_metadata=record.publisher_metadata,
            commercial_metadata=record.commercial_metadata,
            visibility=record.visibility,
        )
        validate_listing_source_consistency(source, listing)
        return listing


_DEFAULT_PROJECTION = DefaultMarketplaceListingProjection()


def project_marketplace_record(
    source: CapabilitySourceIdentity,
    record: MarketplaceListingRecord,
) -> CapabilityCatalogEntry:
    """Map one marketplace row to a federated catalog entry."""
    return _DEFAULT_PROJECTION.project_catalog_entry(source, record)


def build_marketplace_listing(
    source: CapabilitySourceIdentity,
    record: MarketplaceListingRecord,
) -> MarketplaceCapabilityListing:
    """Build a product listing wrapping a canonical catalog entry."""
    return _DEFAULT_PROJECTION.build_listing(source, record)


def default_marketplace_listing_projection() -> MarketplaceListingProjection:
    """Return the platform default listing projection instance."""
    return _DEFAULT_PROJECTION

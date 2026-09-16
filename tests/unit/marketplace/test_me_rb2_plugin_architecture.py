# © Artur Czarnecki. All rights reserved.

"""ME-RB2 plugin architecture proof — external implementations via public contracts."""

from __future__ import annotations

import pytest

from intergrax.capability_catalog import FederatedCapabilityCatalog
from intergrax.contracts.capability_catalog import (
    CapabilityCatalogEntry,
    CapabilityCatalogSource,
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.marketplace import (
    MarketplaceCapabilityListing,
    MarketplaceListingProjection,
    MarketplaceListingRecord,
    MarketplaceMetadataSource,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.marketplace import MarketplaceCatalogService

pytestmark = pytest.mark.unit

_OFFICIAL = CapabilitySourceIdentity(
    source_id="custom.official.marketplace",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


class _CustomCatalogSource:
    @property
    def source_id(self) -> str:
        return _OFFICIAL.source_id

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        return (
            CapabilityCatalogEntry(
                identity=CapabilityDiscoveryIdentity(
                    kind=CapabilityKind.TOOL,
                    source=_OFFICIAL,
                    logical=CapabilityLogicalIdentity(
                        kind=CapabilityKind.TOOL,
                        logical_id="tools.custom.plugin",
                    ),
                ),
                provenance=CapabilityProvenance(
                    source=_OFFICIAL,
                    version_label="9.9.9",
                    publisher="custom-publisher",
                ),
                display_label="Custom Plugin Tool",
            ),
        )


class _CustomListingProjection:
    @property
    def projection_id(self) -> str:
        return "custom.test.projection"

    def project_catalog_entry(
        self,
        source: CapabilitySourceIdentity,
        record: MarketplaceListingRecord,
    ) -> CapabilityCatalogEntry:
        return CapabilityCatalogEntry(
            identity=CapabilityDiscoveryIdentity(
                kind=record.kind,
                source=source,
                logical=CapabilityLogicalIdentity(
                    kind=record.kind,
                    logical_id=record.logical_id,
                ),
            ),
            provenance=CapabilityProvenance(
                source=source,
                version_label=record.version_label,
                publisher=record.publisher,
            ),
            display_label=record.display_label or record.logical_id,
        )

    def build_listing(
        self,
        source: CapabilitySourceIdentity,
        record: MarketplaceListingRecord,
    ) -> MarketplaceCapabilityListing:
        return MarketplaceCapabilityListing(
            listing_id=record.listing_id,
            capability=self.project_catalog_entry(source, record),
            publisher_metadata=record.publisher_metadata,
            commercial_metadata=record.commercial_metadata,
        )


class _CustomMetadataSource:
    def __init__(self) -> None:
        self._projection = _CustomListingProjection()
        self._record = MarketplaceListingRecord(
            kind=CapabilityKind.TOOL,
            logical_id="tools.custom.plugin",
            version_label="9.9.9",
            publisher="custom-publisher",
            display_label="Custom Plugin Tool",
            listing_id="listing-custom-1",
        )

    @property
    def source_id(self) -> str:
        return _OFFICIAL.source_id

    @property
    def source(self) -> CapabilitySourceIdentity:
        return _OFFICIAL

    def read_listings(self) -> tuple[MarketplaceCapabilityListing, ...]:
        return (self._projection.build_listing(self.source, self._record),)


def test_custom_capability_catalog_source_federates_without_subclassing_default() -> None:
    source: CapabilityCatalogSource = _CustomCatalogSource()
    catalog = FederatedCapabilityCatalog((source,))
    snapshot = catalog.snapshot()
    assert len(snapshot.entries) == 1
    assert snapshot.entries[0].identity.logical.logical_id == "tools.custom.plugin"


def test_custom_metadata_source_wires_into_marketplace_service_without_subclassing_default() -> None:
    catalog_source: CapabilityCatalogSource = _CustomCatalogSource()
    metadata_source: MarketplaceMetadataSource = _CustomMetadataSource()
    catalog = FederatedCapabilityCatalog((catalog_source,))
    service = MarketplaceCatalogService(
        catalog=catalog,
        marketplace_sources=(metadata_source,),
    )
    entry = catalog.snapshot().entries[0]
    listing = service.get_listing(
        CapabilityIdentityKey.from_discovery_identity(entry.identity),
    )
    assert listing is not None
    assert listing.listing_id == "listing-custom-1"


def test_custom_listing_projection_builds_listing_without_default_class() -> None:
    projection: MarketplaceListingProjection = _CustomListingProjection()
    record = MarketplaceListingRecord(
        kind=CapabilityKind.TOOL,
        logical_id="tools.projection.only",
        publisher="acme",
    )
    listing = projection.build_listing(_OFFICIAL, record)
    assert listing.capability.provenance.publisher == "acme"

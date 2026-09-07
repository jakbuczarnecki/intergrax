# © Artur Czarnecki. All rights reserved.

"""Stage 11 marketplace canonical join integrity tests."""

from __future__ import annotations

import pytest

from intergrax.capability_catalog import FederatedCapabilityCatalog
from intergrax.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryIdentity,
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.marketplace import MarketplacePublisherMetadata
from intergrax.marketplace import (
    MarketplaceCapabilityCatalogSource,
    MarketplaceCatalogConfigurationError,
    MarketplaceCatalogService,
    MarketplaceListingRecord,
)

pytestmark = pytest.mark.unit

_SOURCE = CapabilitySourceIdentity(
    source_id="official.foo",
    source_kind=CapabilitySourceKind.OFFICIAL,
)
_LOGICAL_ID = "tools.foo"
_IDENTITY_KEY = CapabilityIdentityKey(
    kind=CapabilityKind.TOOL,
    source_id=_SOURCE.source_id,
    source_kind=_SOURCE.source_kind,
    logical_id=_LOGICAL_ID,
)


def _discovery_query(**kwargs: object) -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
        **kwargs,
    )


def _canonical_entry(
    *,
    version_label: str | None = "1.0",
    content_digest: str | None = "AAA",
    publisher: str | None = "intergrax",
    package_reference: str | None = None,
) -> CapabilityCatalogEntry:
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=_SOURCE,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id=_LOGICAL_ID,
            ),
        ),
        provenance=CapabilityProvenance(
            source=_SOURCE,
            version_label=version_label,
            content_digest=content_digest,
            publisher=publisher,
            package_reference=package_reference,
        ),
        display_label="Tool Foo",
    )


class _InconsistentMarketplaceSource(MarketplaceCapabilityCatalogSource):
    """Federation reads canonical facts; product layer exposes divergent listing capability."""

    def __init__(
        self,
        *,
        canonical_entry: CapabilityCatalogEntry,
        listing_capability: CapabilityCatalogEntry,
        record: MarketplaceListingRecord,
    ) -> None:
        super().__init__(source=_SOURCE, records=(record,))
        self._canonical_entry = canonical_entry
        self._listing_capability = listing_capability

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        return (self._canonical_entry,)

    def read_listings(self):
        listing = super().read_listings()[0]
        return (listing.model_copy(update={"capability": self._listing_capability}),)


class _OrphanListingSource(MarketplaceCapabilityCatalogSource):
    """Product metadata exists while federation snapshot omits the identity."""

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        return ()


class _DuplicateListingSource(MarketplaceCapabilityCatalogSource):
    """Malformed product source returning the same identity twice."""

    def read_listings(self):
        listing = super().read_listings()[0]
        return (listing, listing)


class _MutableFederationSource(MarketplaceCapabilityCatalogSource):
    """Federation entries that can change after service construction."""

    def __init__(
        self,
        *,
        source: CapabilitySourceIdentity,
        records: tuple[MarketplaceListingRecord, ...],
        entries: tuple[CapabilityCatalogEntry, ...],
    ) -> None:
        super().__init__(source=source, records=records)
        self._entries = entries

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        return self._entries

    def set_entries(self, entries: tuple[CapabilityCatalogEntry, ...]) -> None:
        self._entries = entries


def _marketplace_record(
    *,
    version_label: str | None = None,
    content_digest: str | None = None,
    publisher: str | None = None,
    package_reference: str | None = None,
) -> MarketplaceListingRecord:
    return MarketplaceListingRecord(
        kind=CapabilityKind.TOOL,
        logical_id=_LOGICAL_ID,
        display_label="Tool Foo",
        version_label=version_label,
        content_digest=content_digest,
        publisher=publisher,
        package_reference=package_reference,
        publisher_metadata=MarketplacePublisherMetadata(
            publisher_id=publisher or "intergrax",
            display_name="Publisher",
        )
        if publisher is not None
        else None,
    )


def test_provenance_mismatch_fails_closed() -> None:
    canonical = _canonical_entry()
    listing_capability = _canonical_entry(
        version_label="2.0",
        content_digest="BBB",
        publisher="vendor",
    )
    marketplace_source = _InconsistentMarketplaceSource(
        canonical_entry=canonical,
        listing_capability=listing_capability,
        record=_marketplace_record(
            version_label="2.0",
            content_digest="BBB",
            publisher="vendor",
        ),
    )
    catalog = FederatedCapabilityCatalog((marketplace_source,))
    with pytest.raises(
        MarketplaceCatalogConfigurationError,
        match="marketplace listing canonical facts must equal federated catalog entry",
    ):
        MarketplaceCatalogService(
            catalog=catalog,
            marketplace_sources=(marketplace_source,),
        )


def test_package_reference_mismatch_fails_closed() -> None:
    canonical = _canonical_entry(package_reference="A")
    listing_capability = _canonical_entry(package_reference="B")
    marketplace_source = _InconsistentMarketplaceSource(
        canonical_entry=canonical,
        listing_capability=listing_capability,
        record=_marketplace_record(package_reference="B"),
    )
    catalog = FederatedCapabilityCatalog((marketplace_source,))
    with pytest.raises(MarketplaceCatalogConfigurationError):
        MarketplaceCatalogService(
            catalog=catalog,
            marketplace_sources=(marketplace_source,),
        )


def test_exact_canonical_match_succeeds() -> None:
    canonical = _canonical_entry()
    marketplace_source = MarketplaceCapabilityCatalogSource(
        source=_SOURCE,
        records=(
            _marketplace_record(
                version_label="1.0",
                content_digest="AAA",
                publisher="intergrax",
            ),
        ),
    )
    catalog = FederatedCapabilityCatalog((marketplace_source,))
    service = MarketplaceCatalogService(
        catalog=catalog,
        marketplace_sources=(marketplace_source,),
    )
    views = service.list_listings(_discovery_query(kinds=(CapabilityKind.TOOL,)))
    assert len(views) == 1
    assert views[0].listing.capability == canonical


def test_marketplace_source_not_in_federation_fails_closed() -> None:
    marketplace_source = MarketplaceCapabilityCatalogSource(
        source=_SOURCE,
        records=(_marketplace_record(),),
    )
    other_source = MarketplaceCapabilityCatalogSource(
        source=CapabilitySourceIdentity(
            source_id="official.other",
            source_kind=CapabilitySourceKind.OFFICIAL,
        ),
        records=(
            MarketplaceListingRecord(
                kind=CapabilityKind.TOOL,
                logical_id="tools.other",
            ),
        ),
    )
    catalog = FederatedCapabilityCatalog((other_source,))
    with pytest.raises(
        MarketplaceCatalogConfigurationError,
        match="marketplace catalog source must be present in federated catalog",
    ):
        MarketplaceCatalogService(
            catalog=catalog,
            marketplace_sources=(marketplace_source,),
        )


def test_get_listing_cannot_bypass_federation() -> None:
    marketplace_source = _OrphanListingSource(
        source=_SOURCE,
        records=(_marketplace_record(),),
    )
    catalog = FederatedCapabilityCatalog((marketplace_source,))
    service = MarketplaceCatalogService(
        catalog=catalog,
        marketplace_sources=(marketplace_source,),
    )
    assert service.get_listing(_IDENTITY_KEY) is None


def test_duplicate_marketplace_source_id_fails_closed() -> None:
    source_a = MarketplaceCapabilityCatalogSource(
        source=_SOURCE,
        records=(_marketplace_record(),),
    )
    source_b = MarketplaceCapabilityCatalogSource(
        source=_SOURCE,
        records=(_marketplace_record(),),
    )
    catalog = FederatedCapabilityCatalog((source_a,))
    with pytest.raises(
        MarketplaceCatalogConfigurationError,
        match="duplicate catalog source_id in federation",
    ):
        MarketplaceCatalogService(
            catalog=catalog,
            marketplace_sources=(source_a, source_b),
        )


def test_duplicate_identity_across_product_sources_fails_closed() -> None:
    marketplace_source = _DuplicateListingSource(
        source=_SOURCE,
        records=(_marketplace_record(),),
    )
    catalog = FederatedCapabilityCatalog((marketplace_source,))
    with pytest.raises(
        MarketplaceCatalogConfigurationError,
        match="duplicate marketplace listing for the same source-qualified discovery identity",
    ):
        MarketplaceCatalogService(
            catalog=catalog,
            marketplace_sources=(marketplace_source,),
        )


def test_get_listing_returns_none_after_federation_removes_capability() -> None:
    canonical = _canonical_entry()
    marketplace_source = _MutableFederationSource(
        source=_SOURCE,
        records=(
            _marketplace_record(
                version_label="1.0",
                content_digest="AAA",
                publisher="intergrax",
            ),
        ),
        entries=(canonical,),
    )
    catalog = FederatedCapabilityCatalog((marketplace_source,))
    service = MarketplaceCatalogService(
        catalog=catalog,
        marketplace_sources=(marketplace_source,),
    )
    assert service.get_listing(_IDENTITY_KEY) is not None

    marketplace_source.set_entries(())
    assert service.get_listing(_IDENTITY_KEY) is None


def test_get_listing_reflects_updated_canonical_facts() -> None:
    canonical_v1 = _canonical_entry(version_label="1.0", content_digest="AAA")
    marketplace_source = _MutableFederationSource(
        source=_SOURCE,
        records=(
            _marketplace_record(
                version_label="1.0",
                content_digest="AAA",
                publisher="intergrax",
            ),
        ),
        entries=(canonical_v1,),
    )
    catalog = FederatedCapabilityCatalog((marketplace_source,))
    service = MarketplaceCatalogService(
        catalog=catalog,
        marketplace_sources=(marketplace_source,),
    )

    canonical_v2 = _canonical_entry(version_label="2.0", content_digest="BBB")
    marketplace_source.set_entries((canonical_v2,))
    listing = service.get_listing(_IDENTITY_KEY)
    assert listing is not None
    assert listing.capability.provenance.version_label == "2.0"
    assert listing.capability.provenance.content_digest == "BBB"
    assert listing.publisher_metadata is not None
    assert listing.publisher_metadata.publisher_id == "intergrax"


def test_list_listings_uses_fresh_federation_snapshot() -> None:
    canonical = _canonical_entry()
    marketplace_source = _MutableFederationSource(
        source=_SOURCE,
        records=(
            _marketplace_record(
                version_label="1.0",
                content_digest="AAA",
                publisher="intergrax",
            ),
        ),
        entries=(canonical,),
    )
    catalog = FederatedCapabilityCatalog((marketplace_source,))
    service = MarketplaceCatalogService(
        catalog=catalog,
        marketplace_sources=(marketplace_source,),
    )
    assert len(service.list_listings(_discovery_query(kinds=(CapabilityKind.TOOL,)))) == 1

    marketplace_source.set_entries(())
    assert service.list_listings(_discovery_query(kinds=(CapabilityKind.TOOL,))) == ()

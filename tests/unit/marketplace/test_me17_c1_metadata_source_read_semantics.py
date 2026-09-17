# © Artur Czarnecki. All rights reserved.

"""ME-17-C1 — metadata source read semantics at query boundary."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

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
    source_id="official.me17c1",
    source_kind=CapabilitySourceKind.OFFICIAL,
)
_LOGICAL_ID = "tools.me17c1.widget"
_IDENTITY_KEY = CapabilityIdentityKey(
    kind=CapabilityKind.TOOL,
    source_id=_SOURCE.source_id,
    source_kind=_SOURCE.source_kind,
    logical_id=_LOGICAL_ID,
)


class _MetadataProviderOperationalError(OSError):
    """Typed operational failure from a metadata provider."""


def _discovery_query(**kwargs: object) -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
        **kwargs,
    )


def _canonical_entry(*, content_digest: str = "AAA") -> CapabilityCatalogEntry:
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
            version_label="1.0",
            content_digest=content_digest,
            publisher="intergrax",
        ),
        display_label="Widget",
    )


def _record(*, listing_id: str | None = "listing-me17c1") -> MarketplaceListingRecord:
    return MarketplaceListingRecord(
        kind=CapabilityKind.TOOL,
        logical_id=_LOGICAL_ID,
        display_label="Widget",
        version_label="1.0",
        content_digest="AAA",
        publisher="intergrax",
        listing_id=listing_id,
        publisher_metadata=MarketplacePublisherMetadata(
            publisher_id="intergrax",
            display_name="Intergrax",
        ),
    )


def _base_source() -> MarketplaceCapabilityCatalogSource:
    return MarketplaceCapabilityCatalogSource(source=_SOURCE, records=(_record(),))


def _wrap_source(
    inner: MarketplaceCapabilityCatalogSource,
    *,
    read_listings,
) -> MarketplaceCapabilityCatalogSource:
    class _Wrapped(MarketplaceCapabilityCatalogSource):
        def __init__(self) -> None:
            self._inner = inner

        @property
        def source_id(self) -> str:
            return self._inner.source_id

        @property
        def source(self) -> CapabilitySourceIdentity:
            return self._inner.source

        def read_listings(self):
            return read_listings(self._inner)

        def read_entries(self):
            return self._inner.read_entries()

    return _Wrapped()


def _service_with_counter() -> tuple[MarketplaceCatalogService, dict[str, int]]:
    counters: dict[str, int] = {"reads": 0}
    inner = _base_source()

    def _read_listings(source: MarketplaceCapabilityCatalogSource):
        counters["reads"] += 1
        return source.read_listings()

    wrapped = _wrap_source(inner, read_listings=_read_listings)
    catalog = FederatedCapabilityCatalog((wrapped,))
    service = MarketplaceCatalogService(catalog=catalog, marketplace_sources=(wrapped,))
    return service, counters


def test_me17_c1_marketplace_service_constructor_does_not_read_metadata_sources() -> None:
    _, counters = _service_with_counter()
    assert counters["reads"] == 0


def test_me17_c1_query_reads_each_metadata_source_once() -> None:
    service, counters = _service_with_counter()
    service.query_listings(_discovery_query(kinds=(CapabilityKind.TOOL,)))
    assert counters["reads"] == 1
    service.query_listings(_discovery_query(kinds=(CapabilityKind.TOOL,)))
    assert counters["reads"] == 2


def test_me17_c1_get_listing_reads_each_metadata_source_once() -> None:
    service, counters = _service_with_counter()
    listing = service.get_listing(_IDENTITY_KEY)
    assert listing is not None
    assert counters["reads"] == 1
    service.get_listing(_IDENTITY_KEY)
    assert counters["reads"] == 2


def test_me17_c1_metadata_provider_failure_occurs_at_query_not_construction() -> None:
    inner = _base_source()
    state = {"fail_next": True}

    def _read_listings(source: MarketplaceCapabilityCatalogSource):
        if state["fail_next"]:
            state["fail_next"] = False
            raise _MetadataProviderOperationalError("provider unavailable")
        return source.read_listings()

    wrapped = _wrap_source(inner, read_listings=_read_listings)
    catalog = FederatedCapabilityCatalog((wrapped,))
    service = MarketplaceCatalogService(catalog=catalog, marketplace_sources=(wrapped,))
    with pytest.raises(_MetadataProviderOperationalError, match="provider unavailable"):
        service.query_listings(_discovery_query(kinds=(CapabilityKind.TOOL,)))
    views = service.query_listings(_discovery_query(kinds=(CapabilityKind.TOOL,))).listing_views
    assert len(views) == 1


def test_me17_c1_programming_provider_failure_propagates() -> None:
    inner = _base_source()

    def _read_listings(_source: MarketplaceCapabilityCatalogSource):
        raise RuntimeError("metadata programming defect")

    wrapped = _wrap_source(inner, read_listings=_read_listings)
    catalog = FederatedCapabilityCatalog((wrapped,))
    service = MarketplaceCatalogService(catalog=catalog, marketplace_sources=(wrapped,))
    with pytest.raises(RuntimeError, match="metadata programming defect"):
        service.get_listing(_IDENTITY_KEY)


def test_me17_c1_duplicate_identity_in_runtime_snapshot_fails_closed() -> None:
    inner = _base_source()

    def _read_listings(source: MarketplaceCapabilityCatalogSource):
        listing = source.read_listings()[0]
        return (listing, listing)

    wrapped = _wrap_source(inner, read_listings=_read_listings)
    service = MarketplaceCatalogService(
        catalog=FederatedCapabilityCatalog((wrapped,)),
        marketplace_sources=(wrapped,),
    )
    with pytest.raises(
        MarketplaceCatalogConfigurationError,
        match="duplicate marketplace listing",
    ):
        service.query_listings(_discovery_query())


def test_me17_c1_canonical_listing_mismatch_fails_closed() -> None:
    canonical = _canonical_entry(content_digest="AAA")
    mismatched = _canonical_entry(content_digest="BBB")

    class _FederatedMismatch(MarketplaceCapabilityCatalogSource):
        def read_entries(self):
            return (canonical,)

        def read_listings(self):
            listing = super().read_listings()[0]
            return (listing.model_copy(update={"capability": mismatched}),)

    source = _FederatedMismatch(source=_SOURCE, records=(_record(),))
    service = MarketplaceCatalogService(
        catalog=FederatedCapabilityCatalog((source,)),
        marketplace_sources=(source,),
    )
    with pytest.raises(
        MarketplaceCatalogConfigurationError,
        match="marketplace listing canonical facts must equal federated catalog entry",
    ):
        service.get_listing(_IDENTITY_KEY)


def test_me17_c1_dynamic_metadata_source_changes_between_queries() -> None:
    inner = _base_source()
    tag = {"n": 0}

    def _read_listings(source: MarketplaceCapabilityCatalogSource):
        tag["n"] += 1
        listing = source.read_listings()[0]
        return (listing.model_copy(update={"listing_id": f"listing-{tag['n']}"}),)

    wrapped = _wrap_source(inner, read_listings=_read_listings)
    service = MarketplaceCatalogService(
        catalog=FederatedCapabilityCatalog((wrapped,)),
        marketplace_sources=(wrapped,),
    )
    first = service.query_listings(_discovery_query(kinds=(CapabilityKind.TOOL,))).listing_views[0]
    second = service.query_listings(_discovery_query(kinds=(CapabilityKind.TOOL,))).listing_views[0]
    assert first.listing.listing_id == "listing-1"
    assert second.listing.listing_id == "listing-2"


def test_me17_c1_failed_metadata_read_does_not_poison_next_query() -> None:
    test_me17_c1_metadata_provider_failure_occurs_at_query_not_construction()


def test_me17_c1_invalid_snapshot_does_not_poison_next_query() -> None:
    inner = _base_source()
    mode = {"duplicate": True}

    def _read_listings(source: MarketplaceCapabilityCatalogSource):
        listing = source.read_listings()[0]
        if mode["duplicate"]:
            mode["duplicate"] = False
            return (listing, listing)
        return (listing,)

    wrapped = _wrap_source(inner, read_listings=_read_listings)
    service = MarketplaceCatalogService(
        catalog=FederatedCapabilityCatalog((wrapped,)),
        marketplace_sources=(wrapped,),
    )
    with pytest.raises(MarketplaceCatalogConfigurationError):
        service.query_listings(_discovery_query())
    views = service.query_listings(_discovery_query(kinds=(CapabilityKind.TOOL,))).listing_views
    assert len(views) == 1


def test_me17_c1_concurrent_queries_do_not_share_mutable_listing_index() -> None:
    inner = _base_source()
    read_counter = {"n": 0}
    counter_lock = threading.Lock()
    start_barrier = threading.Barrier(2)

    def _read_listings(source: MarketplaceCapabilityCatalogSource):
        with counter_lock:
            read_counter["n"] += 1
            tag = read_counter["n"]
        listing = source.read_listings()[0]
        return (listing.model_copy(update={"listing_id": f"read-{tag}"}),)

    wrapped = _wrap_source(inner, read_listings=_read_listings)
    service = MarketplaceCatalogService(
        catalog=FederatedCapabilityCatalog((wrapped,)),
        marketplace_sources=(wrapped,),
    )

    def _run() -> str:
        start_barrier.wait()
        listing = service.get_listing(_IDENTITY_KEY)
        assert listing is not None
        return listing.listing_id or ""

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: _run(), range(2)))
    assert set(results) == {"read-1", "read-2"}


def test_me17_c1_duplicate_source_ids_still_fail_at_construction() -> None:
    source_a = _base_source()
    source_b = _base_source()
    catalog = FederatedCapabilityCatalog((source_a,))
    with pytest.raises(
        MarketplaceCatalogConfigurationError,
        match="duplicate catalog source_id",
    ):
        MarketplaceCatalogService(
            catalog=catalog,
            marketplace_sources=(source_a, source_b),
        )


def test_me17_c1_metadata_source_must_belong_to_catalog_federation() -> None:
    source = _base_source()
    other = MarketplaceCapabilityCatalogSource(
        source=CapabilitySourceIdentity(
            source_id="official.other.me17c1",
            source_kind=CapabilitySourceKind.OFFICIAL,
        ),
        records=(
            MarketplaceListingRecord(
                kind=CapabilityKind.TOOL,
                logical_id="tools.other",
            ),
        ),
    )
    catalog = FederatedCapabilityCatalog((other,))
    with pytest.raises(
        MarketplaceCatalogConfigurationError,
        match="marketplace catalog source must be present in federated catalog",
    ):
        MarketplaceCatalogService(catalog=catalog, marketplace_sources=(source,))

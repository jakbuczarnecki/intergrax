# © Artur Czarnecki. All rights reserved.

"""ME-11-C1 cache observability isolation and marketplace metadata freshness."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.capability_catalog import FederatedCapabilityCatalog, SnapshotCachingCapabilityCatalog
from intergrax.capability_catalog.snapshot import (
    CapabilityCatalogFederationCompleteness,
    CapabilityCatalogSnapshot,
)
from intergrax.capability_catalog.snapshot_cache import build_snapshot_cache_key
from intergrax.contracts.capability_catalog import (
    CapabilityCatalogSnapshotCacheFailurePolicy,
    CapabilityCatalogSnapshotCacheIntegrityError,
    CapabilityCatalogSnapshotCacheKey,
    CapabilityCatalogSnapshotCacheObserverEmitError,
    CapabilityCatalogSnapshotCacheObserverFailurePolicy,
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityKind,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.marketplace import MarketplaceListingRecord, MarketplaceQueryContext
from intergrax.contracts.marketplace.visibility import (
    MarketplaceVisibility,
    MarketplaceVisibilityScope,
)
from intergrax.marketplace import MarketplaceCatalogService
from intergrax.marketplace.projection import DefaultMarketplaceListingProjection

pytestmark = pytest.mark.unit

_OFFICIAL = CapabilitySourceIdentity(
    source_id="official.intergrax.marketplace",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


def _public_record(logical_id: str, *, listing_id: str | None = None) -> MarketplaceListingRecord:
    return MarketplaceListingRecord(
        kind=CapabilityKind.TOOL,
        logical_id=logical_id,
        display_label=logical_id,
        publisher="intergrax",
        listing_id=listing_id,
    )


def _tenant_private_record(logical_id: str, tenant_id: str) -> MarketplaceListingRecord:
    return MarketplaceListingRecord(
        kind=CapabilityKind.TOOL,
        logical_id=logical_id,
        display_label=logical_id,
        publisher="intergrax",
        visibility=MarketplaceVisibility(
            scope=MarketplaceVisibilityScope.TENANT_PRIVATE,
            tenant_id=tenant_id,
        ),
    )


def _org_private_record(logical_id: str, organization_id: str) -> MarketplaceListingRecord:
    return MarketplaceListingRecord(
        kind=CapabilityKind.TOOL,
        logical_id=logical_id,
        display_label=logical_id,
        publisher="intergrax",
        visibility=MarketplaceVisibility(
            scope=MarketplaceVisibilityScope.ORGANIZATION_PRIVATE,
            organization_id=organization_id,
        ),
    )


def _global_query() -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
    )


class _MutableMarketplaceCatalogSource:
    """Federated catalog + metadata with mutable records (test host)."""

    def __init__(self) -> None:
        self._records: tuple[MarketplaceListingRecord, ...] = ()
        self._projection = DefaultMarketplaceListingProjection()

    def set_records(self, records: tuple[MarketplaceListingRecord, ...]) -> None:
        self._records = records

    @property
    def source_id(self) -> str:
        return _OFFICIAL.source_id

    @property
    def source(self) -> CapabilitySourceIdentity:
        return _OFFICIAL

    def read_listings(self):
        return tuple(self._projection.build_listing(_OFFICIAL, record) for record in self._records)

    def read_entries(self):
        entries = [listing.capability for listing in self.read_listings()]
        return tuple(sorted(entries, key=lambda entry: entry.identity.sort_key))


class _TokenGenerationPolicy:
    def __init__(self, token: str) -> None:
        self._token = token

    def generation_tokens(self, sources) -> tuple[str, ...]:
        return (self._token,)


class _CustomSnapshotCache:
    def __init__(self) -> None:
        self.store: dict[CapabilityCatalogSnapshotCacheKey, CapabilityCatalogSnapshot] = {}

    @property
    def cache_id(self) -> str:
        return "custom.me11c1.snapshot_cache"

    def read(self, key: CapabilityCatalogSnapshotCacheKey):
        return self.store.get(key)

    def write(self, key: CapabilityCatalogSnapshotCacheKey, snapshot: CapabilityCatalogSnapshot) -> None:
        self.store[key] = snapshot

    def invalidate(self, key: CapabilityCatalogSnapshotCacheKey) -> None:
        self.store.pop(key, None)


class _FailingCacheObserver:
    def observe(self, **kwargs: object) -> None:
        raise RuntimeError("observer failed")


def _cached_service(
    source: _MutableMarketplaceCatalogSource,
    *,
    cache: _CustomSnapshotCache | None = None,
    generation_policy: _TokenGenerationPolicy | None = None,
    observer: object | None = None,
    observer_failure_policy: CapabilityCatalogSnapshotCacheObserverFailurePolicy = (
        CapabilityCatalogSnapshotCacheObserverFailurePolicy.BEST_EFFORT
    ),
) -> tuple[MarketplaceCatalogService, SnapshotCachingCapabilityCatalog]:
    cache = cache or _CustomSnapshotCache()
    inner = FederatedCapabilityCatalog((source,))
    catalog = SnapshotCachingCapabilityCatalog(
        inner,
        cache=cache,
        generation_policy=generation_policy,
        observer=observer,
        observer_failure_policy=observer_failure_policy,
    )
    service = MarketplaceCatalogService(catalog=catalog, marketplace_sources=(source,))
    return service, catalog


def test_marketplace_service_does_not_pin_listing_index_at_construction() -> None:
    tree = ast.parse(Path("intergrax/marketplace/service.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "__init__":
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute) and target.attr == "_listing_index":
                        raise AssertionError("construction-time _listing_index is forbidden")


def test_cache_observer_failure_does_not_change_hit_result_under_best_effort() -> None:
    source = _MutableMarketplaceCatalogSource()
    source.set_records((_public_record("tool-a"),))
    cache = _CustomSnapshotCache()
    inner = FederatedCapabilityCatalog((source,))
    baseline = SnapshotCachingCapabilityCatalog(inner, cache=cache)
    observed = SnapshotCachingCapabilityCatalog(
        inner,
        cache=cache,
        observer=_FailingCacheObserver(),
    )
    assert baseline.snapshot() == observed.snapshot()
    assert baseline.snapshot() == observed.snapshot()


def test_cache_observer_failure_does_not_change_miss_result_under_best_effort() -> None:
    source = _MutableMarketplaceCatalogSource()
    source.set_records((_public_record("tool-a"),))
    catalog = SnapshotCachingCapabilityCatalog(
        FederatedCapabilityCatalog((source,)),
        cache=_CustomSnapshotCache(),
        observer=_FailingCacheObserver(),
    )
    snap = catalog.snapshot()
    assert len(snap.entries) == 1


def test_cache_observer_failure_does_not_change_write_result_under_best_effort() -> None:
    source = _MutableMarketplaceCatalogSource()
    source.set_records((_public_record("tool-a"),))
    cache = _CustomSnapshotCache()
    catalog = SnapshotCachingCapabilityCatalog(
        FederatedCapabilityCatalog((source,)),
        cache=cache,
        observer=_FailingCacheObserver(),
    )
    snap = catalog.snapshot()
    assert len(cache.store) == 1
    assert snap.entries[0].identity.logical.logical_id == "tool-a"


def test_cache_observer_failure_does_not_block_invalidate_under_best_effort() -> None:
    source = _MutableMarketplaceCatalogSource()
    source.set_records((_public_record("tool-a"),))
    cache = _CustomSnapshotCache()
    catalog = SnapshotCachingCapabilityCatalog(
        FederatedCapabilityCatalog((source,)),
        cache=cache,
        observer=_FailingCacheObserver(),
    )
    catalog.snapshot()
    catalog.invalidate_cached_snapshot()
    assert not cache.store


def test_cache_observer_strict_failure_raises_typed_error() -> None:
    source = _MutableMarketplaceCatalogSource()
    source.set_records((_public_record("tool-a"),))
    catalog = SnapshotCachingCapabilityCatalog(
        FederatedCapabilityCatalog((source,)),
        cache=_CustomSnapshotCache(),
        observer=_FailingCacheObserver(),
        observer_failure_policy=CapabilityCatalogSnapshotCacheObserverFailurePolicy.STRICT,
    )
    with pytest.raises(CapabilityCatalogSnapshotCacheObserverEmitError):
        catalog.snapshot()


def test_cached_partial_snapshot_is_not_accepted_as_valid_hit() -> None:
    source = _MutableMarketplaceCatalogSource()
    source.set_records((_public_record("tool-a"),))
    cache = _CustomSnapshotCache()
    inner = FederatedCapabilityCatalog((source,))
    catalog = SnapshotCachingCapabilityCatalog(inner, cache=cache)
    key = build_snapshot_cache_key(inner.sources)
    authoritative = inner.snapshot()
    partial = authoritative.model_copy(
        update={
            "federation_completeness": CapabilityCatalogFederationCompleteness.PARTIAL,
            "unavailable_source_ids": ("other.missing",),
        },
    )
    cache.store[key] = partial
    result = catalog.snapshot()
    assert result.federation_completeness == CapabilityCatalogFederationCompleteness.COMPLETE


def test_cached_snapshot_with_wrong_source_set_is_not_accepted() -> None:
    source = _MutableMarketplaceCatalogSource()
    source.set_records((_public_record("tool-a"),))
    cache = _CustomSnapshotCache()
    inner = FederatedCapabilityCatalog((source,))
    catalog = SnapshotCachingCapabilityCatalog(inner, cache=cache)
    key = build_snapshot_cache_key(inner.sources)
    wrong = CapabilityCatalogSnapshot(
        source_ids=("wrong.source",),
        entries=inner.snapshot().entries,
    )
    cache.store[key] = wrong
    result = catalog.snapshot()
    assert tuple(result.source_ids) == (_OFFICIAL.source_id,)


def test_corrupt_cache_falls_back_to_authority_under_fallback_policy() -> None:
    source = _MutableMarketplaceCatalogSource()
    source.set_records((_public_record("tool-a"),))
    cache = _CustomSnapshotCache()
    inner = FederatedCapabilityCatalog((source,))
    catalog = SnapshotCachingCapabilityCatalog(
        inner,
        cache=cache,
        cache_failure_policy=CapabilityCatalogSnapshotCacheFailurePolicy.FALLBACK_TO_AUTHORITY,
    )
    key = build_snapshot_cache_key(inner.sources)
    cache.store[key] = CapabilityCatalogSnapshot(source_ids=("wrong.source",), entries=())
    snap = catalog.snapshot()
    assert len(snap.entries) == 1


def test_corrupt_cache_propagates_typed_integrity_error_under_propagate_policy() -> None:
    source = _MutableMarketplaceCatalogSource()
    source.set_records((_public_record("tool-a"),))
    cache = _CustomSnapshotCache()
    inner = FederatedCapabilityCatalog((source,))
    catalog = SnapshotCachingCapabilityCatalog(
        inner,
        cache=cache,
        cache_failure_policy=CapabilityCatalogSnapshotCacheFailurePolicy.PROPAGATE,
    )
    key = build_snapshot_cache_key(inner.sources)
    cache.store[key] = CapabilityCatalogSnapshot(source_ids=("wrong.source",), entries=())
    with pytest.raises(CapabilityCatalogSnapshotCacheIntegrityError):
        catalog.snapshot()


def test_public_to_tenant_private_refresh_does_not_leak_to_foreign_tenant() -> None:
    source = _MutableMarketplaceCatalogSource()
    source.set_records((_public_record("shared-tool"),))
    policy = _TokenGenerationPolicy("gen-1")
    service, catalog = _cached_service(source, generation_policy=policy)
    catalog.snapshot()
    assert len(
        service.list_listings(
            _global_query(),
            marketplace_query_context=MarketplaceQueryContext(tenant_id="tenant-b"),
        ),
    ) == 1
    source.set_records((_tenant_private_record("shared-tool", "tenant-a"),))
    policy._token = "gen-2"
    catalog.snapshot()
    views = service.list_listings(
        _global_query(),
        marketplace_query_context=MarketplaceQueryContext(tenant_id="tenant-b"),
    )
    assert views == ()


def test_public_to_org_private_refresh_does_not_leak_to_foreign_org() -> None:
    source = _MutableMarketplaceCatalogSource()
    source.set_records((_public_record("shared-tool"),))
    policy = _TokenGenerationPolicy("gen-1")
    service, catalog = _cached_service(source, generation_policy=policy)
    catalog.snapshot()
    source.set_records((_org_private_record("shared-tool", "org-a"),))
    policy._token = "gen-2"
    catalog.snapshot()
    views = service.list_listings(
        _global_query(),
        marketplace_query_context=MarketplaceQueryContext(organization_id="org-b"),
    )
    assert views == ()


def test_private_to_public_refresh_is_reflected() -> None:
    source = _MutableMarketplaceCatalogSource()
    source.set_records((_tenant_private_record("shared-tool", "tenant-a"),))
    service, _catalog = _cached_service(source)
    assert service.list_listings(_global_query()) == ()
    source.set_records((_public_record("shared-tool"),))
    views = service.list_listings(_global_query())
    assert len(views) == 1
    assert views[0].listing.capability.identity.logical.logical_id == "shared-tool"


def test_added_listing_after_generation_refresh_becomes_visible() -> None:
    source = _MutableMarketplaceCatalogSource()
    source.set_records((_public_record("tool-a"),))
    policy = _TokenGenerationPolicy("gen-1")
    service, catalog = _cached_service(source, generation_policy=policy)
    catalog.snapshot()
    source.set_records((_public_record("tool-a"), _public_record("tool-b")))
    policy._token = "gen-2"
    catalog.snapshot()
    ids = {v.listing.capability.identity.logical.logical_id for v in service.list_listings(_global_query())}
    assert ids == {"tool-a", "tool-b"}


def test_removed_listing_after_generation_refresh_disappears() -> None:
    source = _MutableMarketplaceCatalogSource()
    source.set_records((_public_record("tool-a"), _public_record("tool-b")))
    policy = _TokenGenerationPolicy("gen-1")
    service, catalog = _cached_service(source, generation_policy=policy)
    catalog.snapshot()
    source.set_records((_public_record("tool-a"),))
    policy._token = "gen-2"
    catalog.snapshot()
    ids = {v.listing.capability.identity.logical.logical_id for v in service.list_listings(_global_query())}
    assert ids == {"tool-a"}


def test_get_listing_uses_current_visibility_after_refresh() -> None:
    source = _MutableMarketplaceCatalogSource()
    source.set_records((_public_record("shared-tool"),))
    service, _catalog = _cached_service(source)
    entry = source.read_entries()[0]
    key = CapabilityIdentityKey.from_discovery_identity(entry.identity)
    assert service.get_listing(key, marketplace_query_context=MarketplaceQueryContext(tenant_id="tenant-b"))
    source.set_records((_tenant_private_record("shared-tool", "tenant-a"),))
    assert (
        service.get_listing(key, marketplace_query_context=MarketplaceQueryContext(tenant_id="tenant-b"))
        is None
    )


def test_marketplace_product_metadata_refreshes_with_current_snapshot() -> None:
    source = _MutableMarketplaceCatalogSource()
    source.set_records((_public_record("shared-tool", listing_id="listing-v1"),))
    service, _catalog = _cached_service(source)
    entry = source.read_entries()[0]
    key = CapabilityIdentityKey.from_discovery_identity(entry.identity)
    first = service.get_listing(key)
    assert first is not None
    assert first.listing_id == "listing-v1"
    source.set_records((_public_record("shared-tool", listing_id="listing-v2"),))
    second = service.get_listing(key)
    assert second is not None
    assert second.listing_id == "listing-v2"


def test_cache_programming_error_still_propagates() -> None:
    class _BrokenCache:
        @property
        def cache_id(self) -> str:
            return "broken"

        def read(self, key):
            raise TypeError("programming defect")

        def write(self, key, snapshot) -> None:
            raise TypeError("programming defect")

        def invalidate(self, key) -> None:
            raise TypeError("programming defect")

    source = _MutableMarketplaceCatalogSource()
    source.set_records((_public_record("tool-a"),))
    catalog = SnapshotCachingCapabilityCatalog(
        FederatedCapabilityCatalog((source,)),
        cache=_BrokenCache(),
    )
    with pytest.raises(TypeError, match="programming defect"):
        catalog.snapshot()

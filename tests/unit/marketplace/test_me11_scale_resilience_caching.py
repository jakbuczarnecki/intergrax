# © Artur Czarnecki. All rights reserved.

"""ME-11 scale, resilience, and snapshot cache proofs."""

from __future__ import annotations

import pytest

from intergrax.capability_catalog import (
    CapabilityCatalogEntry,
    CapabilityCatalogIdentityConflict,
    CapabilityCatalogSourceFailure,
    FederatedCapabilityCatalog,
    SnapshotCachingCapabilityCatalog,
)
from intergrax.capability_catalog.snapshot import CapabilityCatalogFederationCompleteness
from intergrax.capability_catalog.snapshot_cache import BoundedInMemoryCapabilityCatalogSnapshotCache
from intergrax.contracts.capability_catalog import (
    CapabilityCatalogFederationPolicy,
    CapabilityCatalogSnapshotCacheDisposition,
    CapabilityCatalogSnapshotCacheFailurePolicy,
    CapabilityCatalogSnapshotCacheKey,
    CapabilityCatalogSnapshotCacheUnavailableError,
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
from intergrax.contracts.marketplace import MarketplaceListingRecord, MarketplaceQueryContext
from intergrax.contracts.marketplace.visibility import (
    MarketplaceVisibility,
    MarketplaceVisibilityScope,
)
from intergrax.marketplace import MarketplaceCapabilityCatalogSource, MarketplaceCatalogService

pytestmark = pytest.mark.unit

_OFFICIAL = CapabilitySourceIdentity(
    source_id="official.intergrax.marketplace",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


def _entry(source_id: str, logical_id: str) -> CapabilityCatalogEntry:
    source = CapabilitySourceIdentity(
        source_id=source_id,
        source_kind=CapabilitySourceKind.OFFICIAL,
    )
    kind = CapabilityKind.TOOL
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=kind,
            source=source,
            logical=CapabilityLogicalIdentity(kind=kind, logical_id=logical_id),
        ),
        provenance=CapabilityProvenance(source=source, version_label="1.0.0"),
        display_label=logical_id,
    )


class _StaticSource:
    def __init__(
        self,
        source_id: str,
        entries: tuple[CapabilityCatalogEntry, ...],
    ) -> None:
        self._source_id = source_id
        self._entries = entries
        self.read_calls = 0

    @property
    def source_id(self) -> str:
        return self._source_id

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        self.read_calls += 1
        return self._entries


class _FailingSource:
    @property
    def source_id(self) -> str:
        return "zzz.failing"

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        raise CapabilityCatalogSourceFailure("expected outage")


class _CustomSnapshotCache:
    def __init__(self) -> None:
        self.store: dict[CapabilityCatalogSnapshotCacheKey, object] = {}
        self.read_calls = 0
        self.write_calls = 0

    @property
    def cache_id(self) -> str:
        return "custom.me11.snapshot_cache"

    def read(self, key: CapabilityCatalogSnapshotCacheKey):
        self.read_calls += 1
        return self.store.get(key)

    def write(self, key: CapabilityCatalogSnapshotCacheKey, snapshot) -> None:
        self.write_calls += 1
        self.store[key] = snapshot

    def invalidate(self, key: CapabilityCatalogSnapshotCacheKey) -> None:
        self.store.pop(key, None)


class _ObservingCache(_CustomSnapshotCache):
    def __init__(self, observer: list[CapabilityCatalogSnapshotCacheDisposition]) -> None:
        super().__init__()
        self._observer = observer

    def read(self, key: CapabilityCatalogSnapshotCacheKey):
        result = super().read(key)
        self._observer.append(
            CapabilityCatalogSnapshotCacheDisposition.HIT
            if result is not None
            else CapabilityCatalogSnapshotCacheDisposition.MISS,
        )
        return result


class _UnavailableCache:
    @property
    def cache_id(self) -> str:
        return "unavailable.me11"

    def read(self, key: CapabilityCatalogSnapshotCacheKey):
        raise CapabilityCatalogSnapshotCacheUnavailableError("backend down")

    def write(self, key: CapabilityCatalogSnapshotCacheKey, snapshot) -> None:
        raise CapabilityCatalogSnapshotCacheUnavailableError("backend down")

    def invalidate(self, key: CapabilityCatalogSnapshotCacheKey) -> None:
        raise CapabilityCatalogSnapshotCacheUnavailableError("backend down")


class _BrokenCache:
    @property
    def cache_id(self) -> str:
        return "broken.me11"

    def read(self, key: CapabilityCatalogSnapshotCacheKey):
        raise TypeError("programming defect")


class _TokenGenerationPolicy:
    def __init__(self, token: str) -> None:
        self._token = token

    def generation_tokens(self, sources) -> tuple[str, ...]:
        return (self._token,)


def _marketplace_service_with_cache(
    cache: object,
    *,
    records: tuple[MarketplaceListingRecord, ...] | None = None,
) -> tuple[MarketplaceCatalogService, _StaticSource]:
    recs = records or (
        MarketplaceListingRecord(
            kind=CapabilityKind.TOOL,
            logical_id="public-tool",
            display_label="public",
            publisher="intergrax",
        ),
        MarketplaceListingRecord(
            kind=CapabilityKind.TOOL,
            logical_id="tenant-a-private",
            display_label="private",
            publisher="intergrax",
            visibility=MarketplaceVisibility(
                scope=MarketplaceVisibilityScope.TENANT_PRIVATE,
                tenant_id="tenant-a",
            ),
        ),
    )
    source = MarketplaceCapabilityCatalogSource(source=_OFFICIAL, records=recs)
    inner = FederatedCapabilityCatalog((source,))
    catalog = SnapshotCachingCapabilityCatalog(inner, cache=cache)
    service = MarketplaceCatalogService(catalog=catalog, marketplace_sources=(source,))
    return service, source


def _global_query() -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
    )


def test_custom_marketplace_cache_plugs_in_without_core_changes() -> None:
    source = _StaticSource("src.a", (_entry("src.a", "tool.one"),))
    cache = _CustomSnapshotCache()
    catalog = SnapshotCachingCapabilityCatalog(FederatedCapabilityCatalog((source,)), cache=cache)
    catalog.snapshot()
    assert cache.write_calls == 1
    source.read_calls = 0
    catalog.snapshot()
    assert source.read_calls == 0
    assert cache.read_calls == 2


def test_cache_hit_is_semantically_equivalent_to_uncached_result() -> None:
    source = _StaticSource("src.a", (_entry("src.a", "tool.one"),))
    inner = FederatedCapabilityCatalog((source,))
    cache = BoundedInMemoryCapabilityCatalogSnapshotCache(max_entries=8)
    catalog = SnapshotCachingCapabilityCatalog(inner, cache=cache)
    first = catalog.snapshot()
    second = catalog.snapshot()
    assert first == second
    assert source.read_calls == 1


def test_cache_failure_falls_back_to_authoritative_path_under_configured_policy() -> None:
    source = _StaticSource("src.a", (_entry("src.a", "tool.one"),))
    catalog = SnapshotCachingCapabilityCatalog(
        FederatedCapabilityCatalog((source,)),
        cache=_UnavailableCache(),
        cache_failure_policy=CapabilityCatalogSnapshotCacheFailurePolicy.FALLBACK_TO_AUTHORITY,
    )
    snapshot = catalog.snapshot()
    assert len(snapshot.entries) == 1
    assert source.read_calls == 1


def test_cache_does_not_bypass_tenant_visibility() -> None:
    cache = BoundedInMemoryCapabilityCatalogSnapshotCache(max_entries=4)
    service, _ = _marketplace_service_with_cache(cache)
    tenant_a = MarketplaceQueryContext(tenant_id="tenant-a")
    tenant_b = MarketplaceQueryContext(tenant_id="tenant-b")
    a_views = service.list_listings(_global_query(), marketplace_query_context=tenant_a)
    b_views = service.list_listings(_global_query(), marketplace_query_context=tenant_b)
    a_ids = {v.listing.capability.identity.logical.logical_id for v in a_views}
    b_ids = {v.listing.capability.identity.logical.logical_id for v in b_views}
    assert "tenant-a-private" in a_ids
    assert "tenant-a-private" not in b_ids


def test_cache_does_not_bypass_organization_visibility() -> None:
    records = (
        MarketplaceListingRecord(
            kind=CapabilityKind.SKILL,
            logical_id="org-private",
            display_label="org",
            publisher="intergrax",
            visibility=MarketplaceVisibility(
                scope=MarketplaceVisibilityScope.ORGANIZATION_PRIVATE,
                organization_id="org-a",
            ),
        ),
    )
    cache = BoundedInMemoryCapabilityCatalogSnapshotCache(max_entries=4)
    service, _ = _marketplace_service_with_cache(cache, records=records)
    allowed = service.list_listings(
        _global_query(),
        marketplace_query_context=MarketplaceQueryContext(organization_id="org-a"),
    )
    denied = service.list_listings(
        _global_query(),
        marketplace_query_context=MarketplaceQueryContext(organization_id="org-b"),
    )
    assert len(allowed) == 1
    assert denied == ()


def test_cached_snapshot_preserves_release_provenance() -> None:
    entry = _entry("src.a", "tool.provenance")
    source = _StaticSource("src.a", (entry,))
    catalog = SnapshotCachingCapabilityCatalog(
        FederatedCapabilityCatalog((source,)),
        cache=BoundedInMemoryCapabilityCatalogSnapshotCache(max_entries=2),
    )
    cached = catalog.snapshot()
    assert cached.entries[0].provenance == entry.provenance
    assert cached.entries[0].provenance.version_label == "1.0.0"


def test_invalid_snapshot_is_not_cached() -> None:
    base = _entry("official.catalog", "tools.conflict")
    conflicting = _entry("official.catalog", "tools.conflict",).model_copy(
        update={"display_label": "Different label"},
    )

    class _ConflictingSource:
        @property
        def source_id(self) -> str:
            return "official.catalog"

        def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
            return (base, conflicting)

    cache = _CustomSnapshotCache()
    catalog = SnapshotCachingCapabilityCatalog(
        FederatedCapabilityCatalog((_ConflictingSource(),)),
        cache=cache,
    )
    with pytest.raises(CapabilityCatalogIdentityConflict):
        catalog.snapshot()
    assert cache.write_calls == 0


def test_snapshot_change_invalidates_or_misses_old_cache() -> None:
    mutable = _StaticSource("src.a", (_entry("src.a", "v1"),))
    inner = FederatedCapabilityCatalog((mutable,))
    policy_v1 = _TokenGenerationPolicy("gen-1")
    cache = BoundedInMemoryCapabilityCatalogSnapshotCache(max_entries=4)
    catalog = SnapshotCachingCapabilityCatalog(
        inner,
        cache=cache,
        generation_policy=policy_v1,
    )
    first = catalog.snapshot()
    policy_v1._token = "gen-2"
    catalog_gen2 = SnapshotCachingCapabilityCatalog(
        inner,
        cache=cache,
        generation_policy=policy_v1,
    )
    mutable._entries = (_entry("src.a", "v2"),)
    second = catalog_gen2.snapshot()
    assert first.entries[0].identity.logical.logical_id == "v1"
    assert second.entries[0].identity.logical.logical_id == "v2"
    assert mutable.read_calls == 2


def test_unexpected_cache_programming_error_is_not_silently_normalized() -> None:
    catalog = SnapshotCachingCapabilityCatalog(
        FederatedCapabilityCatalog((_StaticSource("src.a", (_entry("src.a", "x"),)),)),
        cache=_BrokenCache(),
    )
    with pytest.raises(TypeError, match="programming defect"):
        catalog.snapshot()


def test_expected_source_failure_follows_explicit_resilience_policy() -> None:
    federated = FederatedCapabilityCatalog(
        (_StaticSource("aaa.ok", (_entry("aaa.ok", "ok"),)), _FailingSource()),
    )
    with pytest.raises(CapabilityCatalogSourceFailure):
        federated.snapshot(
            federation_policy=CapabilityCatalogFederationPolicy.STRICT_COMPLETE,
        )
    partial = federated.snapshot(
        federation_policy=CapabilityCatalogFederationPolicy.ALLOW_PARTIAL,
    )
    assert partial.federation_completeness == CapabilityCatalogFederationCompleteness.PARTIAL
    assert partial.unavailable_source_ids == ("zzz.failing",)


def test_unexpected_source_programming_error_propagates() -> None:
    class _BuggySource:
        @property
        def source_id(self) -> str:
            return "buggy"

        def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
            raise RuntimeError("defect")

    with pytest.raises(RuntimeError, match="defect"):
        FederatedCapabilityCatalog((_BuggySource(),)).snapshot()


def test_partial_snapshot_is_never_reported_as_complete() -> None:
    partial = FederatedCapabilityCatalog(
        (_StaticSource("aaa.ok", (_entry("aaa.ok", "ok"),)), _FailingSource()),
    ).snapshot(federation_policy=CapabilityCatalogFederationPolicy.ALLOW_PARTIAL)
    assert partial.federation_completeness != CapabilityCatalogFederationCompleteness.COMPLETE


def test_source_failure_never_widens_private_visibility() -> None:
    public = _StaticSource("public.src", (_entry("public.src", "public-tool"),))

    class _PrivateFailingSource:
        @property
        def source_id(self) -> str:
            return "private.src"

        def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
            raise CapabilityCatalogSourceFailure("private catalog unavailable")

    partial = FederatedCapabilityCatalog((public, _PrivateFailingSource())).snapshot(
        federation_policy=CapabilityCatalogFederationPolicy.ALLOW_PARTIAL,
    )
    assert partial.unavailable_source_ids == ("private.src",)
    assert len(partial.entries) == 1
    assert partial.entries[0].identity.logical.logical_id == "public-tool"


def test_many_sources_merge_deterministically() -> None:
    sources = tuple(
        _StaticSource(f"src.{index:03d}", (_entry(f"src.{index:03d}", f"tool.{index}"),))
        for index in range(40)
    )
    snapshot = FederatedCapabilityCatalog(sources).snapshot()
    assert len(snapshot.entries) == 40
    sort_keys = [entry.identity.sort_key for entry in snapshot.entries]
    assert sort_keys == sorted(sort_keys)


def test_large_catalog_preserves_unique_identity_and_order() -> None:
    entries = tuple(
        _entry("bulk.source", f"tool.{index}") for index in range(2500)
    )
    snapshot = FederatedCapabilityCatalog((_StaticSource("bulk.source", entries),)).snapshot()
    keys = [entry.identity.sort_key for entry in snapshot.entries]
    assert len(keys) == len(set(keys))
    assert keys == sorted(keys)


def test_source_order_does_not_change_canonical_snapshot() -> None:
    a = _StaticSource("aaa.first", (_entry("aaa.first", "one"),))
    b = _StaticSource("bbb.second", (_entry("bbb.second", "two"),))
    first = FederatedCapabilityCatalog((a, b)).snapshot()
    second = FederatedCapabilityCatalog((b, a)).snapshot()
    assert first == second


def test_bounded_cache_eviction_does_not_break_correctness() -> None:
    source = _StaticSource("src.a", (_entry("src.a", "tool.one"),))
    cache = BoundedInMemoryCapabilityCatalogSnapshotCache(max_entries=1)
    key_b = CapabilityCatalogSnapshotCacheKey.for_federation(
        source_ids=("other.source",),
        generation_tokens=("x",),
    )
    cache.write(key_b, FederatedCapabilityCatalog((source,)).snapshot())
    catalog = SnapshotCachingCapabilityCatalog(FederatedCapabilityCatalog((source,)), cache=cache)
    snap = catalog.snapshot()
    assert len(snap.entries) == 1


def test_observability_cache_events_do_not_change_result() -> None:
    events: list[CapabilityCatalogSnapshotCacheDisposition] = []
    source = _StaticSource("src.a", (_entry("src.a", "x"),))
    catalog = SnapshotCachingCapabilityCatalog(
        FederatedCapabilityCatalog((source,)),
        cache=_ObservingCache(events),
    )
    first = catalog.snapshot()
    second = catalog.snapshot()
    assert first == second
    assert CapabilityCatalogSnapshotCacheDisposition.MISS in events
    assert CapabilityCatalogSnapshotCacheDisposition.HIT in events

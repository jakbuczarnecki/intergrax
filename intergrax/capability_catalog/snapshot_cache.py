# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reference snapshot cache plugins (ME-11)."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Final

from intergrax.capability_catalog.snapshot import (
    CapabilityCatalogFederationCompleteness,
    CapabilityCatalogSnapshot,
)
from intergrax.capability_catalog.snapshot_cache_port import CapabilityCatalogSnapshotCache
from intergrax.capability_catalog.snapshot_provider import CapabilityCatalogSnapshotProvider
from intergrax.contracts.capability_catalog.federation_policy import (
    CapabilityCatalogFederationPolicy,
)
from intergrax.contracts.capability_catalog.snapshot_cache import (
    NOOP_CAPABILITY_CATALOG_SNAPSHOT_CACHE_ID,
    CapabilityCatalogSnapshotCacheDisposition,
    CapabilityCatalogSnapshotCacheFailurePolicy,
    CapabilityCatalogSnapshotCacheGenerationPolicy,
    CapabilityCatalogSnapshotCacheKey,
    CapabilityCatalogSnapshotCacheObserver,
    CapabilityCatalogSnapshotCacheUnavailableError,
)
from intergrax.contracts.capability_catalog.source import CapabilityCatalogSource

IN_MEMORY_CAPABILITY_CATALOG_SNAPSHOT_CACHE_ID: Final = (
    "capability_catalog.snapshot_cache.in_memory"
)


class _SourceIdsGenerationPolicy:
    """Default cache key material — federation composition only."""

    def generation_tokens(
        self,
        sources: tuple[CapabilityCatalogSource, ...],
    ) -> tuple[str, ...]:
        return ()


class NoOpCapabilityCatalogSnapshotCache:
    """Always miss — correctness path without materialized reads."""

    @property
    def cache_id(self) -> str:
        return NOOP_CAPABILITY_CATALOG_SNAPSHOT_CACHE_ID

    def read(self, key: CapabilityCatalogSnapshotCacheKey) -> CapabilityCatalogSnapshot | None:
        return None

    def write(
        self,
        key: CapabilityCatalogSnapshotCacheKey,
        snapshot: CapabilityCatalogSnapshot,
    ) -> None:
        return None

    def invalidate(self, key: CapabilityCatalogSnapshotCacheKey) -> None:
        return None


class BoundedInMemoryCapabilityCatalogSnapshotCache:
    """Thread-safe bounded in-memory snapshot cache (reference / test host)."""

    def __init__(self, *, max_entries: int) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: OrderedDict[CapabilityCatalogSnapshotCacheKey, CapabilityCatalogSnapshot] = (
            OrderedDict()
        )

    @property
    def cache_id(self) -> str:
        return IN_MEMORY_CAPABILITY_CATALOG_SNAPSHOT_CACHE_ID

    def read(self, key: CapabilityCatalogSnapshotCacheKey) -> CapabilityCatalogSnapshot | None:
        with self._lock:
            value = self._entries.get(key)
            if value is None:
                return None
            self._entries.move_to_end(key)
            return value

    def write(
        self,
        key: CapabilityCatalogSnapshotCacheKey,
        snapshot: CapabilityCatalogSnapshot,
    ) -> None:
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
            self._entries[key] = snapshot
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def invalidate(self, key: CapabilityCatalogSnapshotCacheKey) -> None:
        with self._lock:
            self._entries.pop(key, None)


def build_snapshot_cache_key(
    sources: tuple[CapabilityCatalogSource, ...],
    *,
    generation_policy: CapabilityCatalogSnapshotCacheGenerationPolicy | None = None,
) -> CapabilityCatalogSnapshotCacheKey:
    policy = generation_policy or _SourceIdsGenerationPolicy()
    source_ids = tuple(source.source_id for source in sources)
    return CapabilityCatalogSnapshotCacheKey.for_federation(
        source_ids=source_ids,
        generation_tokens=policy.generation_tokens(sources),
    )


class SnapshotCachingCapabilityCatalog:
    """Decorator over a snapshot provider with optional federated snapshot cache."""

    def __init__(
        self,
        inner: CapabilityCatalogSnapshotProvider,
        *,
        cache: CapabilityCatalogSnapshotCache | None = None,
        cache_failure_policy: CapabilityCatalogSnapshotCacheFailurePolicy = (
            CapabilityCatalogSnapshotCacheFailurePolicy.FALLBACK_TO_AUTHORITY
        ),
        generation_policy: CapabilityCatalogSnapshotCacheGenerationPolicy | None = None,
        observer: CapabilityCatalogSnapshotCacheObserver | None = None,
        federation_policy: CapabilityCatalogFederationPolicy = (
            CapabilityCatalogFederationPolicy.STRICT_COMPLETE
        ),
    ) -> None:
        self._inner = inner
        self._cache = cache or NoOpCapabilityCatalogSnapshotCache()
        self._cache_failure_policy = cache_failure_policy
        self._generation_policy = generation_policy
        self._observer = observer
        self._federation_policy = federation_policy

    @property
    def sources(self) -> tuple[CapabilityCatalogSource, ...]:
        return self._inner.sources

    def snapshot(
        self,
        *,
        federation_policy: CapabilityCatalogFederationPolicy | None = None,
    ) -> CapabilityCatalogSnapshot:
        policy = federation_policy or self._federation_policy
        key = build_snapshot_cache_key(self._inner.sources, generation_policy=self._generation_policy)
        cache_id = self._cache.cache_id
        try:
            cached = self._cache.read(key)
        except CapabilityCatalogSnapshotCacheUnavailableError:
            self._observe(
                CapabilityCatalogSnapshotCacheDisposition.UNAVAILABLE,
                cache_id=cache_id,
            )
            if (
                self._cache_failure_policy
                == CapabilityCatalogSnapshotCacheFailurePolicy.FALLBACK_TO_AUTHORITY
            ):
                return self._inner.snapshot(federation_policy=policy)
            raise
        except Exception:
            raise

        if cached is not None:
            self._observe(
                CapabilityCatalogSnapshotCacheDisposition.HIT,
                cache_id=cache_id,
                entry_count=len(cached.entries),
            )
            return cached

        self._observe(CapabilityCatalogSnapshotCacheDisposition.MISS, cache_id=cache_id)
        snapshot = self._inner.snapshot(federation_policy=policy)
        if snapshot.federation_completeness != CapabilityCatalogFederationCompleteness.COMPLETE:
            return snapshot
        try:
            self._cache.write(key, snapshot)
        except CapabilityCatalogSnapshotCacheUnavailableError:
            self._observe(
                CapabilityCatalogSnapshotCacheDisposition.UNAVAILABLE,
                cache_id=cache_id,
            )
            if (
                self._cache_failure_policy
                == CapabilityCatalogSnapshotCacheFailurePolicy.FALLBACK_TO_AUTHORITY
            ):
                return snapshot
            raise
        except Exception:
            raise
        self._observe(
            CapabilityCatalogSnapshotCacheDisposition.WRITE,
            cache_id=cache_id,
            entry_count=len(snapshot.entries),
        )
        return snapshot

    def invalidate_cached_snapshot(self) -> None:
        key = build_snapshot_cache_key(
            self._inner.sources,
            generation_policy=self._generation_policy,
        )
        try:
            self._cache.invalidate(key)
        except CapabilityCatalogSnapshotCacheUnavailableError:
            if (
                self._cache_failure_policy
                == CapabilityCatalogSnapshotCacheFailurePolicy.PROPAGATE
            ):
                raise
        except Exception:
            raise
        self._observe(
            CapabilityCatalogSnapshotCacheDisposition.INVALIDATE,
            cache_id=self._cache.cache_id,
        )

    def _observe(
        self,
        disposition: CapabilityCatalogSnapshotCacheDisposition,
        *,
        cache_id: str,
        entry_count: int | None = None,
    ) -> None:
        if self._observer is None:
            return
        self._observer.observe(
            disposition=disposition,
            cache_id=cache_id,
            entry_count=entry_count,
        )

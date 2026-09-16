# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Federated snapshot cache SPI (ME-11)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.capability_catalog.snapshot import CapabilityCatalogSnapshot
from intergrax.contracts.capability_catalog.snapshot_cache import (
    CapabilityCatalogSnapshotCacheKey,
)


@runtime_checkable
class CapabilityCatalogSnapshotCache(Protocol):
    """Optional materialized read for federated snapshots — not authority."""

    @property
    def cache_id(self) -> str:
        """Stable plugin identifier (not a Python class name)."""

    def read(self, key: CapabilityCatalogSnapshotCacheKey) -> CapabilityCatalogSnapshot | None:
        """Return a cached snapshot or ``None`` on miss."""

    def write(
        self,
        key: CapabilityCatalogSnapshotCacheKey,
        snapshot: CapabilityCatalogSnapshot,
    ) -> None:
        """Store a validated snapshot as a whole entry."""

    def invalidate(self, key: CapabilityCatalogSnapshotCacheKey) -> None:
        """Drop one cache entry when present."""

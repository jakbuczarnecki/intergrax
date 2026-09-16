# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pluggable federated catalog snapshot cache contracts (ME-11)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.source import CapabilityCatalogSource

SCHEMA_CAPABILITY_CATALOG_SNAPSHOT_CACHE_KEY_V1: Final = (
    "capability_catalog_snapshot_cache_key.v1"
)
NOOP_CAPABILITY_CATALOG_SNAPSHOT_CACHE_ID: Final = (
    "capability_catalog.snapshot_cache.noop"
)


class CapabilityCatalogSnapshotCacheFailurePolicy(StrEnum):
    """When the cache port fails unexpectedly at the infrastructure boundary."""

    FALLBACK_TO_AUTHORITY = "fallback_to_authority"
    PROPAGATE = "propagate"


class CapabilityCatalogSnapshotCacheObserverFailurePolicy(StrEnum):
    """When a cache lifecycle observer raises — observational only (ME-10)."""

    BEST_EFFORT = "best_effort"
    STRICT = "strict"


class CapabilityCatalogSnapshotCacheDisposition(StrEnum):
    HIT = "hit"
    MISS = "miss"
    STALE = "stale"
    WRITE = "write"
    INVALIDATE = "invalidate"
    UNAVAILABLE = "unavailable"


class CapabilityCatalogSnapshotCacheKey(BaseModel):
    """Semantically complete cache identity for a federated snapshot projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_catalog_snapshot_cache_key.v1"] = (
        SCHEMA_CAPABILITY_CATALOG_SNAPSHOT_CACHE_KEY_V1
    )
    namespace: str = Field(min_length=1, default="capability_catalog.federated_snapshot")
    federation_source_ids: tuple[str, ...]
    generation_tokens: tuple[str, ...] = ()

    @classmethod
    def for_federation(
        cls,
        *,
        source_ids: tuple[str, ...],
        generation_tokens: tuple[str, ...] = (),
        namespace: str = "capability_catalog.federated_snapshot",
    ) -> CapabilityCatalogSnapshotCacheKey:
        ordered_sources = tuple(sorted(source_ids))
        return cls(
            namespace=require_non_empty_text(namespace, label="namespace"),
            federation_source_ids=ordered_sources,
            generation_tokens=generation_tokens,
        )


@runtime_checkable
class CapabilityCatalogSnapshotCacheGenerationPolicy(Protocol):
    """Derives generation material included in snapshot cache keys."""

    def generation_tokens(
        self,
        sources: tuple[CapabilityCatalogSource, ...],
    ) -> tuple[str, ...]:
        """Stable refresh tokens per federation composition (may be empty)."""


class CapabilityCatalogSnapshotCacheUnavailableError(OSError):
    """Expected operational cache backend failure — not a programming defect."""


class CapabilityCatalogSnapshotCacheIntegrityError(OSError):
    """Cached snapshot failed integrity validation — untrusted cache materialization."""


class CapabilityCatalogSnapshotCacheObserverEmitError(OSError):
    """Cache observer failed under STRICT observer failure policy."""


@runtime_checkable
class CapabilityCatalogSnapshotCacheObserver(Protocol):
    """Observational cache facts — must not influence snapshot results (ME-10)."""

    def observe(
        self,
        *,
        disposition: CapabilityCatalogSnapshotCacheDisposition,
        cache_id: str,
        entry_count: int | None = None,
    ) -> None:
        """Record a cache lifecycle fact without sensitive key material."""



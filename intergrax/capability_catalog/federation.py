# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Federated capability catalog read model (CAPABILITY-CATALOG-1 Stage 2)."""

from __future__ import annotations

from intergrax.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.capability_catalog.errors import (
    CapabilityCatalogConfigurationError,
    CapabilityCatalogIdentityConflict,
    CapabilityCatalogSourceFailure,
)
from intergrax.capability_catalog.snapshot import CapabilityCatalogSnapshot
from intergrax.capability_catalog.source import CapabilityCatalogSource


def _source_id_sort_key(source_id: str) -> str:
    return source_id


def _validate_sources(
    sources: tuple[CapabilityCatalogSource, ...],
) -> tuple[CapabilityCatalogSource, ...]:
    if not sources:
        raise CapabilityCatalogConfigurationError(
            "federated capability catalog requires at least one source",
        )
    seen: set[str] = set()
    for source in sources:
        source_id = source.source_id
        if source_id in seen:
            raise CapabilityCatalogConfigurationError(
                f"duplicate catalog source_id in federation: {source_id!r}",
            )
        seen.add(source_id)
    return sources


def merge_capability_catalog_entries(
    observations: tuple[tuple[str, CapabilityCatalogEntry], ...],
) -> tuple[CapabilityCatalogEntry, ...]:
    """Union observations with exact-duplicate dedupe and conflict fail-closed.

    Follows AC-4 ``merge_federated_candidates`` semantics adapted for catalog entries.
    """
    grouped: dict[
        tuple[str, str, str, str],
        tuple[CapabilityCatalogEntry, list[str]],
    ] = {}
    for source_id, entry in observations:
        identity_key = entry.identity.sort_key
        existing = grouped.get(identity_key)
        if existing is None:
            grouped[identity_key] = (entry, [source_id])
            continue
        canonical_entry, source_ids = existing
        if canonical_entry != entry:
            raise CapabilityCatalogIdentityConflict(
                "conflicting catalog facts for the same source-qualified discovery identity",
            )
        source_ids.append(source_id)

    ordered_keys = sorted(grouped.keys())
    return tuple(grouped[identity_key][0] for identity_key in ordered_keys)


class FederatedCapabilityCatalog:
    """Compose multiple read-only catalog sources into one deterministic snapshot."""

    def __init__(self, sources: tuple[CapabilityCatalogSource, ...]) -> None:
        self._sources = _validate_sources(sources)

    @property
    def sources(self) -> tuple[CapabilityCatalogSource, ...]:
        return self._sources

    def snapshot(self) -> CapabilityCatalogSnapshot:
        ordered_sources = tuple(
            sorted(self._sources, key=lambda source: _source_id_sort_key(source.source_id)),
        )
        observations: list[tuple[str, CapabilityCatalogEntry]] = []
        for source in ordered_sources:
            source_id = source.source_id
            try:
                entries = source.read_entries()
            except Exception as exc:
                raise CapabilityCatalogSourceFailure(
                    f"catalog source {source_id!r} failed during read",
                ) from exc
            for entry in entries:
                entry_source_id = entry.identity.source.source_id
                if entry_source_id != source_id:
                    raise CapabilityCatalogConfigurationError(
                        "catalog source "
                        f"{source_id!r} returned entry with mismatched identity source "
                        f"{entry_source_id!r}",
                    )
                observations.append((source_id, entry))

        merged = merge_capability_catalog_entries(tuple(observations))
        ordered_entries = tuple(sorted(merged, key=lambda entry: entry.identity.sort_key))
        return CapabilityCatalogSnapshot(
            source_ids=tuple(
                _source_id_sort_key(source.source_id) for source in ordered_sources
            ),
            entries=ordered_entries,
        )

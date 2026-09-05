# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Immutable federated capability catalog snapshot (CAPABILITY-CATALOG-1 Stage 2)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from intergrax.capability_catalog.entry import CapabilityCatalogEntry

SCHEMA_CAPABILITY_CATALOG_SNAPSHOT_V1: Final = "capability_catalog_snapshot.v1"


class CapabilityCatalogSnapshot(BaseModel):
    """Deterministic, read-only, non-authoritative federated catalog view."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_catalog_snapshot.v1"] = (
        SCHEMA_CAPABILITY_CATALOG_SNAPSHOT_V1
    )
    source_ids: tuple[str, ...]
    entries: tuple[CapabilityCatalogEntry, ...]

    @model_validator(mode="after")
    def _validate_deterministic_ordering(self) -> CapabilityCatalogSnapshot:
        ordered_entries = tuple(sorted(self.entries, key=_entry_sort_key))
        if ordered_entries != self.entries:
            raise ValueError("snapshot entries must be in deterministic identity order")
        ordered_sources = tuple(sorted(self.source_ids))
        if ordered_sources != self.source_ids:
            raise ValueError("snapshot source_ids must be sorted")
        return self


def _entry_sort_key(entry: CapabilityCatalogEntry) -> tuple[str, str, str, str]:
    return entry.identity.sort_key

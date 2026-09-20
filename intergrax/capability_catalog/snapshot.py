# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Immutable federated capability catalog snapshot (CAPABILITY-CATALOG-1 Stage 2)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.contracts.capability_catalog.federation import (
    CapabilityCatalogFederationCompleteness,
)

SCHEMA_CAPABILITY_CATALOG_SNAPSHOT_V1: Final = "capability_catalog_snapshot.v1"

# Re-export canonical vocabulary for existing runtime import paths.
__all__ = [
    "CapabilityCatalogFederationCompleteness",
    "CapabilityCatalogSnapshot",
    "SCHEMA_CAPABILITY_CATALOG_SNAPSHOT_V1",
]


class CapabilityCatalogSnapshot(BaseModel):
    """Deterministic, read-only, non-authoritative federated catalog view."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_catalog_snapshot.v1"] = (
        SCHEMA_CAPABILITY_CATALOG_SNAPSHOT_V1
    )
    source_ids: tuple[str, ...]
    entries: tuple[CapabilityCatalogEntry, ...]
    federation_completeness: CapabilityCatalogFederationCompleteness = (
        CapabilityCatalogFederationCompleteness.COMPLETE
    )
    unavailable_source_ids: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _validate_deterministic_ordering(self) -> CapabilityCatalogSnapshot:
        ordered_entries = tuple(sorted(self.entries, key=_entry_sort_key))
        if ordered_entries != self.entries:
            raise ValueError("snapshot entries must be in deterministic identity order")
        ordered_sources = tuple(sorted(self.source_ids))
        if ordered_sources != self.source_ids:
            raise ValueError("snapshot source_ids must be sorted")
        ordered_unavailable = tuple(sorted(self.unavailable_source_ids))
        if ordered_unavailable != self.unavailable_source_ids:
            raise ValueError("unavailable_source_ids must be sorted")
        if (
            self.federation_completeness
            == CapabilityCatalogFederationCompleteness.COMPLETE
            and self.unavailable_source_ids
        ):
            raise ValueError("complete snapshot cannot list unavailable sources")
        if (
            self.federation_completeness
            == CapabilityCatalogFederationCompleteness.PARTIAL
            and not self.unavailable_source_ids
        ):
            raise ValueError("partial snapshot must list unavailable sources")
        return self


def _entry_sort_key(entry: CapabilityCatalogEntry) -> tuple[str, str, str, str]:
    return entry.identity.sort_key

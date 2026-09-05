# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Read-only capability catalog source port (CAPABILITY-CATALOG-1 Stage 2)."""

from __future__ import annotations

from typing import Protocol

from intergrax.capability_catalog.entry import CapabilityCatalogEntry


class CapabilityCatalogSource(Protocol):
    """Minimal read-only port: source identity → enumerate entries."""

    @property
    def source_id(self) -> str:
        """Stable catalog source instance identifier."""

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        """Return all entries currently visible from this source."""

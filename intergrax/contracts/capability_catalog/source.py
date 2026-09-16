# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Read-only capability catalog source port (CAPABILITY-CATALOG-1 Stage 2)."""

from __future__ import annotations

from typing import Protocol

from intergrax.contracts.capability_catalog.entry import CapabilityCatalogEntry


class CapabilityCatalogSource(Protocol):
    """Minimal read-only port: stable source identity → enumerate catalog entries.

    Failures from ``read_entries`` must propagate to the federation caller; sources
    must not swallow errors or return partial data after a structural failure.
    """

    @property
    def source_id(self) -> str:
        """Stable catalog source instance identifier."""

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        """Return all entries currently visible from this source.

        A source snapshot must not contain conflicting catalog entries for the
        same discovery identity (same kind, source, and logical_id). Federation
        enforces this fail-closed via ``CapabilityCatalogIdentityConflict``.
        """

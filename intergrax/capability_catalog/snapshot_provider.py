# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Federated catalog snapshot provider port (ME-11)."""

from __future__ import annotations

from typing import Protocol

from intergrax.capability_catalog.snapshot import CapabilityCatalogSnapshot
from intergrax.contracts.capability_catalog.federation_policy import (
    CapabilityCatalogFederationPolicy,
)
from intergrax.contracts.capability_catalog.source import CapabilityCatalogSource


class CapabilityCatalogSnapshotProvider(Protocol):
    """Read-only federated snapshot surface — cacheable, composable."""

    @property
    def sources(self) -> tuple[CapabilityCatalogSource, ...]:
        """Configured federation sources."""

    def snapshot(
        self,
        *,
        federation_policy: CapabilityCatalogFederationPolicy = (
            CapabilityCatalogFederationPolicy.STRICT_COMPLETE
        ),
    ) -> CapabilityCatalogSnapshot:
        """Materialize the current federated catalog snapshot."""

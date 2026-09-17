# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace product metadata source SPI (ME-RB2)."""

from __future__ import annotations

from typing import Protocol

from intergrax.contracts.capability_catalog.identity import CapabilitySourceIdentity
from intergrax.contracts.marketplace.listing import MarketplaceCapabilityListing


class MarketplaceMetadataSource(Protocol):
    """Read-only product/listing metadata for one marketplace catalog source.

    Distinct from ``CapabilityCatalogSource``: this port exposes marketplace
    product representation (listings), not generic federated discovery facts.
    Implementations may also satisfy ``CapabilityCatalogSource`` when the same
    backend supplies canonical catalog entries for federation.

    ``read_listings`` is a **runtime snapshot read** (current provider state).
    Callers such as ``MarketplaceCatalogService`` invoke it at query/read
    boundaries, not during service construction. Construction validates static
    wiring (duplicate ``source_id``, federation membership) only.

    ``read_listings`` failures and malformed snapshots must propagate; providers
    must not return empty tuples to mask duplicate identity or identity mismatch.
    """

    @property
    def source_id(self) -> str:
        """Stable marketplace metadata source instance identifier."""

    @property
    def source(self) -> CapabilitySourceIdentity:
        """Declared source identity for listings from this provider."""

    def read_listings(self) -> tuple[MarketplaceCapabilityListing, ...]:
        """Return the current product listing snapshot from this provider."""

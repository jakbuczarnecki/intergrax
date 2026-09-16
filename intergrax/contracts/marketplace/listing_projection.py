# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace listing projection SPI (ME-RB2)."""

from __future__ import annotations

from typing import Protocol

from intergrax.contracts.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.contracts.capability_catalog.identity import CapabilitySourceIdentity
from intergrax.contracts.marketplace.listing import MarketplaceCapabilityListing
from intergrax.contracts.marketplace.listing_record import MarketplaceListingRecord


class MarketplaceListingProjection(Protocol):
    """Maps marketplace product rows to canonical catalog entries and listings.

    Implementations must preserve capability identity, provenance, publisher
    consistency, version/digest fields, and deterministic output for deterministic
    input. They must not mutate governance, ranking, or runtime state.

    Configuration or consistency violations must fail closed (raise); silent
    omission of invalid rows is forbidden at this boundary.
    """

    @property
    def projection_id(self) -> str:
        """Stable projection strategy identifier."""

    def project_catalog_entry(
        self,
        source: CapabilitySourceIdentity,
        record: MarketplaceListingRecord,
    ) -> CapabilityCatalogEntry:
        """Project one marketplace row to federated catalog facts."""

    def build_listing(
        self,
        source: CapabilitySourceIdentity,
        record: MarketplaceListingRecord,
    ) -> MarketplaceCapabilityListing:
        """Build a product listing wrapping projected catalog facts."""

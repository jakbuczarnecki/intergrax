# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Immutable marketplace catalog snapshot (Stage 11)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from intergrax.contracts.capability_catalog.identity import CapabilitySourceIdentity
from intergrax.marketplace.listing import MarketplaceCapabilityListing
from intergrax.marketplace.validation import validate_unique_listings

SCHEMA_MARKETPLACE_CATALOG_SNAPSHOT_V1: Final = "marketplace_catalog_snapshot.v1"


class MarketplaceCatalogSnapshot(BaseModel):
    """In-memory marketplace product snapshot — read-only, no persistence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["marketplace_catalog_snapshot.v1"] = (
        SCHEMA_MARKETPLACE_CATALOG_SNAPSHOT_V1
    )
    source: CapabilitySourceIdentity
    listings: tuple[MarketplaceCapabilityListing, ...]

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        validate_unique_listings(self.listings)

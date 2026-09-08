# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Input records for in-memory marketplace catalog sources (Stage 11)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.marketplace.commercial import MarketplaceCommercialMetadata
from intergrax.contracts.marketplace.publisher import MarketplacePublisherMetadata


@dataclass(frozen=True, slots=True)
class MarketplaceListingRecord:
    """Read-only marketplace row used to build catalog entries and product metadata."""

    kind: CapabilityKind
    logical_id: str
    version_label: str | None = None
    package_reference: str | None = None
    content_digest: str | None = None
    publisher: str | None = None
    display_label: str | None = None
    listing_id: str | None = None
    publisher_metadata: MarketplacePublisherMetadata | None = None
    commercial_metadata: MarketplaceCommercialMetadata | None = None

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Validation helpers for marketplace catalog sources (Stage 11)."""

from __future__ import annotations

from intergrax.contracts.capability_catalog.identity import (
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.marketplace.errors import MarketplaceCatalogConfigurationError
from intergrax.marketplace.listing import MarketplaceCapabilityListing

_MARKETPLACE_SOURCE_KINDS: frozenset[CapabilitySourceKind] = frozenset(
    {
        CapabilitySourceKind.OFFICIAL,
        CapabilitySourceKind.ENTERPRISE_PRIVATE,
        CapabilitySourceKind.THIRD_PARTY,
    },
)


def validate_marketplace_source(source: CapabilitySourceIdentity) -> None:
    """Reject source kinds that marketplace adapters do not represent."""
    if source.source_kind not in _MARKETPLACE_SOURCE_KINDS:
        raise MarketplaceCatalogConfigurationError(
            "marketplace catalog source requires source_kind "
            f"in {sorted(item.value for item in _MARKETPLACE_SOURCE_KINDS)}, "
            f"got {source.source_kind.value!r}",
        )


def validate_unique_listings(
    listings: tuple[MarketplaceCapabilityListing, ...],
) -> None:
    """Fail closed on duplicate discovery identities within one marketplace snapshot."""
    seen: set[tuple[str, str, str, str]] = set()
    for listing in listings:
        identity_key = listing.capability.identity.sort_key
        if identity_key in seen:
            raise MarketplaceCatalogConfigurationError(
                "duplicate marketplace listing for the same source-qualified discovery identity",
            )
        seen.add(identity_key)


def validate_listing_source_consistency(
    source: CapabilitySourceIdentity,
    listing: MarketplaceCapabilityListing,
) -> None:
    """Ensure listing entries match the declared marketplace source identity."""
    entry_source = listing.capability.identity.source
    if entry_source != source:
        raise MarketplaceCatalogConfigurationError(
            "marketplace listing source must match adapter source identity",
        )
    if listing.capability.provenance.source != source:
        raise MarketplaceCatalogConfigurationError(
            "marketplace listing provenance source must match adapter source identity",
        )

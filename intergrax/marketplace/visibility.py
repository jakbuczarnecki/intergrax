# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Central marketplace visibility evaluation (ME-9)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.contracts.marketplace.query_context import MarketplaceQueryContext
from intergrax.contracts.marketplace.visibility import (
    MarketplaceVisibility,
    MarketplaceVisibilityScope,
)

DEFAULT_PUBLIC_MARKETPLACE_VISIBILITY = MarketplaceVisibility(
    scope=MarketplaceVisibilityScope.PUBLIC,
)


def resolve_marketplace_visibility(
    visibility: MarketplaceVisibility | None,
) -> MarketplaceVisibility:
    """Backward-compatible default: listings without visibility are PUBLIC."""
    if visibility is None:
        return DEFAULT_PUBLIC_MARKETPLACE_VISIBILITY
    return visibility


def hard_marketplace_tenant_isolation(
    listing_visibility: MarketplaceVisibility,
    query_context: MarketplaceQueryContext,
) -> bool:
    """Non-disableable tenant isolation floor — foreign private listings are denied."""
    resolved = resolve_marketplace_visibility(listing_visibility)
    if resolved.scope is MarketplaceVisibilityScope.PUBLIC:
        return True
    caller_tenant = query_context.tenant_id
    if caller_tenant is None:
        return False
    return caller_tenant == resolved.tenant_id


class MarketplaceVisibilityPolicyExtension(Protocol):
    """Optional additional restriction within the hard tenant isolation floor."""

    @property
    def policy_id(self) -> str:
        """Stable identifier for diagnostics and tests."""

    def allow(
        self,
        listing_visibility: MarketplaceVisibility,
        query_context: MarketplaceQueryContext,
    ) -> bool:
        """Return False to deny visibility; may not widen beyond hard isolation."""


@dataclass(frozen=True, slots=True)
class MarketplaceVisibilityEvaluator:
    """Single canonical visibility boundary for list/search/lookup/recommendation."""

    extension: MarketplaceVisibilityPolicyExtension | None = None

    def is_visible(
        self,
        listing_visibility: MarketplaceVisibility | None,
        query_context: MarketplaceQueryContext,
    ) -> bool:
        resolved = resolve_marketplace_visibility(listing_visibility)
        if not hard_marketplace_tenant_isolation(resolved, query_context):
            return False
        if self.extension is None:
            return True
        return self.extension.allow(resolved, query_context)

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace listing text search strategy (ME-5)."""

from __future__ import annotations

from typing import Final

from intergrax.capability_catalog.default_text_search import (
    DefaultCatalogEntryTextSearchStrategy,
)

MARKETPLACE_LISTING_TEXT_SEARCH_STRATEGY_ID: Final = "marketplace.listing_text"


class DefaultMarketplaceListingTextSearchStrategy(DefaultCatalogEntryTextSearchStrategy):
    """Default marketplace product filter — same semantics as legacy inline search."""

    @property
    def search_strategy_id(self) -> str:
        return MARKETPLACE_LISTING_TEXT_SEARCH_STRATEGY_ID

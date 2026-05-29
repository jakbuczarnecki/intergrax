# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Web / enterprise search integration contract (§7.1.2, Phase M.2)."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from intergrax.websearch.schemas.search_hit import SearchHit


@runtime_checkable
class SearchProvider(Protocol):
    """
    Provider-agnostic search facade.

    Richer providers may also implement ``intergrax.websearch.providers.base.WebSearchProvider``;
    this contract is the integration-catalog surface for Tier-3 composition.
    """

    def search(self, query: str, *, limit: int = 10) -> Sequence[SearchHit]:
        """Execute search and return ranked hits."""

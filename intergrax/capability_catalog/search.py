# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pluggable capability search over Stage-3 discovery candidates (ME-5)."""

from __future__ import annotations

from typing import Protocol

from intergrax.capability_catalog.candidate import CapabilityDiscoveryCandidate
from intergrax.capability_catalog.search_validation import validate_searched_output
from intergrax.capability_catalog.searched_candidate import SearchedCapabilityCandidate
from intergrax.contracts.capability_catalog.search import (
    CapabilitySearchContext,
    CapabilitySearchQuery,
)


class CapabilitySearchStrategy(Protocol):
    """Structural search plugin — filtering only, never governance or lifecycle."""

    @property
    def search_strategy_id(self) -> str:
        """Stable search strategy identifier."""

    def search(
        self,
        candidates: tuple[CapabilityDiscoveryCandidate, ...],
        query: CapabilitySearchQuery,
        context: CapabilitySearchContext,
    ) -> tuple[SearchedCapabilityCandidate, ...]:
        """Return a filtered candidate subset with search evidence."""


def search_capability_candidates(
    candidates: tuple[CapabilityDiscoveryCandidate, ...],
    strategy: CapabilitySearchStrategy,
    *,
    query: CapabilitySearchQuery | None = None,
    context: CapabilitySearchContext | None = None,
) -> tuple[SearchedCapabilityCandidate, ...]:
    """Run a search strategy and enforce output integrity fail-closed."""
    search_query = query or CapabilitySearchQuery()
    search_context = context or CapabilitySearchContext()
    searched = strategy.search(candidates, search_query, search_context)
    validate_searched_output(
        input_candidates=candidates,
        searched=searched,
        search_strategy_id=strategy.search_strategy_id,
    )
    return searched

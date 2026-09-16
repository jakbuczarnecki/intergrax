# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Fail-closed search output validation (ME-5)."""

from __future__ import annotations

from intergrax.capability_catalog.candidate import CapabilityDiscoveryCandidate
from intergrax.capability_catalog.errors import CapabilitySearchError
from intergrax.capability_catalog.searched_candidate import SearchedCapabilityCandidate
from intergrax.contracts.capability_catalog._validation import require_non_empty_text


def validate_searched_output(
    *,
    input_candidates: tuple[CapabilityDiscoveryCandidate, ...],
    searched: tuple[SearchedCapabilityCandidate, ...],
    search_strategy_id: str,
) -> None:
    """Reject search output that mutates, duplicates, or re-identifies candidates."""
    require_non_empty_text(search_strategy_id, label="search_strategy_id")
    input_by_key = {
        candidate.identity.sort_key: candidate for candidate in input_candidates
    }
    seen_keys: set[tuple[str, str, str, str]] = set()
    last_input_index = -1

    for item in searched:
        if item.evidence.search_strategy_id != search_strategy_id:
            raise CapabilitySearchError(
                "search output evidence.search_strategy_id must match the active strategy",
            )
        key = item.candidate.identity.sort_key
        if key not in input_by_key:
            raise CapabilitySearchError(
                "search output contains unknown candidate identity",
            )
        if key in seen_keys:
            raise CapabilitySearchError(
                "search output contains duplicate candidate identity",
            )
        seen_keys.add(key)

        original = input_by_key[key]
        if item.candidate != original:
            raise CapabilitySearchError(
                "search output must not mutate candidate identity, provenance, or availability",
            )

        input_index = next(
            index
            for index, candidate in enumerate(input_candidates)
            if candidate.identity.sort_key == key
        )
        if input_index < last_input_index:
            raise CapabilitySearchError(
                "search output must preserve input candidate order",
            )
        last_input_index = input_index

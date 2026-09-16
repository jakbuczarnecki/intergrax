# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic catalog-entry text search baseline (ME-5)."""

from __future__ import annotations

from typing import Final

from intergrax.capability_catalog.candidate import CapabilityDiscoveryCandidate
from intergrax.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.capability_catalog.searched_candidate import SearchedCapabilityCandidate
from intergrax.contracts.capability_catalog.search import (
    CapabilitySearchContext,
    CapabilitySearchEvidence,
    CapabilitySearchQuery,
    CapabilitySearchSignal,
)

CATALOG_ENTRY_TEXT_SEARCH_STRATEGY_ID: Final = "catalog.entry_text"


def normalize_search_text(text: str | None) -> str | None:
    if text is None:
        return None
    normalized = text.strip().casefold()
    if not normalized:
        return None
    return normalized


def catalog_entry_matches_text(entry: CapabilityCatalogEntry, query_text: str) -> str | None:
    """Return matched field name when query is a substring of a searchable field."""
    haystacks = (
        ("logical_id", entry.identity.logical.logical_id.casefold()),
        ("display_label", (entry.display_label or "").casefold()),
    )
    for field_name, haystack in haystacks:
        if query_text in haystack:
            return field_name
    return None


class DefaultCatalogEntryTextSearchStrategy:
    """Substring search over canonical catalog entry text fields."""

    @property
    def search_strategy_id(self) -> str:
        return CATALOG_ENTRY_TEXT_SEARCH_STRATEGY_ID

    def search(
        self,
        candidates: tuple[CapabilityDiscoveryCandidate, ...],
        query: CapabilitySearchQuery,
        context: CapabilitySearchContext,
    ) -> tuple[SearchedCapabilityCandidate, ...]:
        del context
        normalized = normalize_search_text(query.text)
        if normalized is None:
            return tuple(
                SearchedCapabilityCandidate(
                    candidate=candidate,
                    evidence=CapabilitySearchEvidence(
                        search_strategy_id=self.search_strategy_id,
                        signal=CapabilitySearchSignal.PASS_THROUGH,
                    ),
                )
                for candidate in candidates
            )

        matched: list[SearchedCapabilityCandidate] = []
        for candidate in candidates:
            matched_field = catalog_entry_matches_text(
                candidate.catalog_entry,
                normalized,
            )
            if matched_field is None:
                continue
            matched.append(
                SearchedCapabilityCandidate(
                    candidate=candidate,
                    evidence=CapabilitySearchEvidence(
                        search_strategy_id=self.search_strategy_id,
                        signal=CapabilitySearchSignal.TEXT_SUBSTRING_MATCH,
                        matched_field=matched_field,
                    ),
                ),
            )
        return tuple(matched)

# © Artur Czarnecki. All rights reserved.

"""ME-5 capability search tests."""

from __future__ import annotations

import pytest

from intergrax.capability_catalog import (
    CapabilityCatalogEntry,
    CapabilityDiscoveryCandidate,
    CapabilitySearchError,
    DefaultCatalogEntryTextSearchStrategy,
    SearchedCapabilityCandidate,
    search_capability_candidates,
)
from intergrax.capability_catalog.default_text_search import CATALOG_ENTRY_TEXT_SEARCH_STRATEGY_ID
from intergrax.contracts.capability_catalog import (
    AvailabilityDisposition,
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilitySearchContext,
    CapabilitySearchEvidence,
    CapabilitySearchQuery,
    CapabilitySearchSignal,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)

pytestmark = pytest.mark.unit


def _candidate(
    *,
    kind: CapabilityKind,
    logical_id: str,
    display_label: str | None = None,
) -> CapabilityDiscoveryCandidate:
    source = CapabilitySourceIdentity(
        source_id="official.catalog",
        source_kind=CapabilitySourceKind.OFFICIAL,
    )
    entry = CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=kind,
            source=source,
            logical=CapabilityLogicalIdentity(kind=kind, logical_id=logical_id),
        ),
        provenance=CapabilityProvenance(source=source),
        display_label=display_label or logical_id,
    )
    return CapabilityDiscoveryCandidate(
        catalog_entry=entry,
        availability=AvailabilityDisposition.CATALOG_AVAILABLE,
    )


def test_default_text_search_filters_by_substring_preserving_order() -> None:
    agent = _candidate(kind=CapabilityKind.AGENT, logical_id="agents.finder", display_label="Finder")
    tool = _candidate(kind=CapabilityKind.TOOL, logical_id="tools.other", display_label="Other")
    skill = _candidate(kind=CapabilityKind.SKILL, logical_id="skills.find.pack", display_label="Pack")
    strategy = DefaultCatalogEntryTextSearchStrategy()
    searched = search_capability_candidates(
        (tool, agent, skill),
        strategy,
        query=CapabilitySearchQuery(text="find"),
    )
    assert [item.candidate.identity.logical.logical_id for item in searched] == [
        "agents.finder",
        "skills.find.pack",
    ]
    assert all(
        item.evidence.search_strategy_id == CATALOG_ENTRY_TEXT_SEARCH_STRATEGY_ID
        for item in searched
    )


def test_default_text_search_empty_query_passes_through() -> None:
    candidates = (_candidate(kind=CapabilityKind.TOOL, logical_id="tools.a"),)
    searched = search_capability_candidates(
        candidates,
        DefaultCatalogEntryTextSearchStrategy(),
        query=CapabilitySearchQuery(text="   "),
    )
    assert len(searched) == 1
    assert searched[0].evidence.signal is CapabilitySearchSignal.PASS_THROUGH


def test_search_output_integrity_rejects_unknown_identity() -> None:
    candidate = _candidate(kind=CapabilityKind.TOOL, logical_id="tools.a")

    class _BrokenSearch:
        @property
        def search_strategy_id(self) -> str:
            return "broken.search"

        def search(self, candidates, query, context):
            del query, context
            return (
                SearchedCapabilityCandidate(
                    candidate=candidate,
                    evidence=CapabilitySearchEvidence(
                        search_strategy_id="wrong.id",
                        signal=CapabilitySearchSignal.PASS_THROUGH,
                    ),
                ),
            )

    with pytest.raises(CapabilitySearchError, match="search_strategy_id"):
        search_capability_candidates((candidate,), _BrokenSearch())


def test_unexpected_programming_error_propagates_from_strategy() -> None:
    class _ExplodingSearch:
        @property
        def search_strategy_id(self) -> str:
            return "explode"

        def search(self, candidates, query, context):
            raise RuntimeError("programming defect")

    with pytest.raises(RuntimeError, match="programming defect"):
        search_capability_candidates(
            (_candidate(kind=CapabilityKind.TOOL, logical_id="tools.a"),),
            _ExplodingSearch(),
            query=CapabilitySearchQuery(text="a"),
            context=CapabilitySearchContext(),
        )

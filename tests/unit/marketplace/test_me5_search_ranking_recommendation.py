# © Artur Czarnecki. All rights reserved.

"""ME-5 search, ranking, and recommendation pluginability proofs."""

from __future__ import annotations

import pytest

from intergrax.capability_catalog import (
    CapabilityDiscoveryCandidate,
    FederatedCapabilityCatalog,
    RankedCapabilityCandidate,
    SearchedCapabilityCandidate,
    StableIdentityRanker,
    rank_capability_candidates,
)
from intergrax.contracts.capability_catalog import (
    AvailabilityDisposition,
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityKind,
    CapabilityRankingEvidence,
    CapabilityRankingSignal,
    CapabilityRecommendationContext,
    CapabilityRecommendationEvidence,
    CapabilityRecommendationReasonCode,
    CapabilitySearchEvidence,
    CapabilitySearchQuery,
    CapabilitySearchSignal,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.marketplace import MarketplaceListingRecord
from intergrax.marketplace import (
    MarketplaceCapabilityCatalogSource,
    MarketplaceCatalogService,
    MarketplaceDiscoveryService,
)
from intergrax.capability_catalog.recommended_capability import CapabilityRecommendation

pytestmark = pytest.mark.unit

_OFFICIAL = CapabilitySourceIdentity(
    source_id="official.intergrax.marketplace",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


def _discovery_query(**kwargs: object) -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
        **kwargs,
    )


def _mixed_catalog_bundle() -> tuple[FederatedCapabilityCatalog, MarketplaceCapabilityCatalogSource]:
    records = (
        MarketplaceListingRecord(
            kind=CapabilityKind.AGENT,
            logical_id="agents.me5.alpha",
            display_label="Alpha Agent",
            publisher="intergrax",
        ),
        MarketplaceListingRecord(
            kind=CapabilityKind.TOOL,
            logical_id="tools.me5.beta",
            display_label="Beta Tool",
            publisher="intergrax",
        ),
        MarketplaceListingRecord(
            kind=CapabilityKind.SKILL,
            logical_id="skills.me5.gamma",
            display_label="Gamma Skill",
            publisher="intergrax",
        ),
    )
    source = MarketplaceCapabilityCatalogSource(source=_OFFICIAL, records=records)
    catalog = FederatedCapabilityCatalog((source,))
    return catalog, source


def _mixed_catalog_service() -> MarketplaceCatalogService:
    catalog, source = _mixed_catalog_bundle()
    return MarketplaceCatalogService(catalog=catalog, marketplace_sources=(source,))


class _AgentOnlySearch:
    @property
    def search_strategy_id(self) -> str:
        return "custom.agent_only"

    def search(self, candidates, query, context):
        del query, context
        return tuple(
            SearchedCapabilityCandidate(
                candidate=candidate,
                evidence=CapabilitySearchEvidence(
                    search_strategy_id=self.search_strategy_id,
                    signal=CapabilitySearchSignal.PASS_THROUGH,
                ),
            )
            for candidate in candidates
            if candidate.identity.kind is CapabilityKind.AGENT
        )


class _ReverseRanker:
    @property
    def ranker_id(self) -> str:
        return "custom.reverse_rank"

    def rank(self, candidates, context):
        del context
        ordered = tuple(reversed(candidates))
        return tuple(
            RankedCapabilityCandidate(
                candidate=candidate,
                evidence=CapabilityRankingEvidence(
                    ranker_id=self.ranker_id,
                    rank_position=index,
                    signal=CapabilityRankingSignal.STABLE_IDENTITY_ORDER,
                ),
            )
            for index, candidate in enumerate(ordered, start=1)
        )


class _SingleRecommendation:
    @property
    def recommendation_strategy_id(self) -> str:
        return "custom.single_pick"

    def recommend(self, ranked, context):
        del context
        pick = ranked[-1]
        return (
            CapabilityRecommendation(
                ranked=pick,
                evidence=CapabilityRecommendationEvidence(
                    recommendation_strategy_id=self.recommendation_strategy_id,
                    reason_codes=(CapabilityRecommendationReasonCode.TOP_RANKED,),
                    reason_text="custom last-ranked pick",
                    rank_position=pick.evidence.rank_position,
                ),
            ),
        )


def test_custom_search_strategy_changes_listing_order_without_core_edit() -> None:
    catalog, source = _mixed_catalog_bundle()
    default_service = MarketplaceCatalogService(catalog=catalog, marketplace_sources=(source,))
    custom_service = MarketplaceCatalogService(
        catalog=catalog,
        marketplace_sources=(source,),
        listing_text_search=_AgentOnlySearch(),
    )
    default_views = default_service.list_listings(_discovery_query())
    custom_views = custom_service.list_listings(_discovery_query())
    assert len(default_views) == 3
    assert len(custom_views) == 1
    assert custom_views[0].listing.capability.identity.kind is CapabilityKind.AGENT


def test_custom_ranker_orders_deterministically() -> None:
    service = _mixed_catalog_service()
    snapshot = service._catalog.snapshot()
    candidates = tuple(
        CapabilityDiscoveryCandidate(
            catalog_entry=entry,
            availability=AvailabilityDisposition.CATALOG_AVAILABLE,
        )
        for entry in snapshot.entries
    )
    ranked = rank_capability_candidates(candidates, _ReverseRanker())
    assert ranked[0].candidate.identity.logical.logical_id == "tools.me5.beta"


def test_custom_recommendation_strategy_via_discovery_service() -> None:
    service = _mixed_catalog_service()
    snapshot = service._catalog.snapshot()
    candidates = tuple(
        CapabilityDiscoveryCandidate(
            catalog_entry=entry,
            availability=AvailabilityDisposition.CATALOG_AVAILABLE,
        )
        for entry in snapshot.entries
    )
    pipeline = MarketplaceDiscoveryService(
        search_strategy=_AgentOnlySearch(),
        ranker=StableIdentityRanker(),
        recommendation_strategy=_SingleRecommendation(),
    )
    recommendations = pipeline.discover_recommendations(
        candidates,
        recommendation_context=CapabilityRecommendationContext(top_n=5),
    )
    assert len(recommendations) == 1
    assert recommendations[0].evidence.recommendation_strategy_id == "custom.single_pick"
    assert recommendations[0].identity.kind is CapabilityKind.AGENT


def test_mixed_agent_tool_skill_search_and_listings() -> None:
    service = _mixed_catalog_service()
    views = service.list_listings(_discovery_query(), query_text="gamma")
    assert len(views) == 1
    assert views[0].listing.capability.identity.kind is CapabilityKind.SKILL


def test_marketplace_discovery_defaults_are_deterministic() -> None:
    service = _mixed_catalog_service()
    snapshot = service._catalog.snapshot()
    candidates = tuple(
        CapabilityDiscoveryCandidate(
            catalog_entry=entry,
            availability=AvailabilityDisposition.CATALOG_AVAILABLE,
        )
        for entry in snapshot.entries
    )
    pipeline = MarketplaceDiscoveryService.with_defaults()
    first = pipeline.discover_recommendations(
        candidates,
        search_query=CapabilitySearchQuery(text="me5"),
    )
    second = pipeline.discover_recommendations(
        candidates,
        search_query=CapabilitySearchQuery(text="me5"),
    )
    assert first == second

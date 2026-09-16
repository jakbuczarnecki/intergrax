# © Artur Czarnecki. All rights reserved.

"""ME-9 multi-tenant / private marketplace isolation proofs."""

from __future__ import annotations

import pytest

from intergrax.capability_catalog import (
    AvailabilityPreservingGovernanceEvaluator,
    FederatedCapabilityCatalog,
    govern_capability_candidates,
    rank_capability_candidates,
)
from intergrax.capability_catalog.candidate import CapabilityDiscoveryCandidate
from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityGovernanceContext,
    CapabilityGovernancePosture,
    CapabilityKind,
    CapabilityRankingContext,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.marketplace import (
    CommercialModel,
    MarketplaceCommercialMetadata,
    MarketplaceListingRecord,
    MarketplaceQueryContext,
    MarketplaceVisibility,
    MarketplaceVisibilityScope,
)
from intergrax.marketplace import (
    MarketplaceCapabilityCatalogSource,
    MarketplaceCatalogService,
    MarketplaceDiscoveryService,
    MarketplaceRecommendationService,
    MarketplaceVisibilityEvaluator,
    hard_marketplace_tenant_isolation,
)
from intergrax.contracts.marketplace.visibility import MarketplaceVisibility as Visibility

pytestmark = pytest.mark.unit

_OFFICIAL = CapabilitySourceIdentity(
    source_id="official.intergrax.marketplace",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


def _public_record(kind: CapabilityKind, logical_id: str) -> MarketplaceListingRecord:
    return MarketplaceListingRecord(
        kind=kind,
        logical_id=logical_id,
        display_label=logical_id,
        publisher="intergrax",
    )


def _private_record(
    kind: CapabilityKind,
    logical_id: str,
    tenant_id: str,
) -> MarketplaceListingRecord:
    return MarketplaceListingRecord(
        kind=kind,
        logical_id=logical_id,
        display_label=logical_id,
        publisher="intergrax",
        visibility=MarketplaceVisibility(
            scope=MarketplaceVisibilityScope.TENANT_PRIVATE,
            tenant_id=tenant_id,
        ),
    )


def _me9_service() -> MarketplaceCatalogService:
    records = (
        _public_record(CapabilityKind.AGENT, "public-agent"),
        _public_record(CapabilityKind.TOOL, "public-tool"),
        _private_record(CapabilityKind.AGENT, "tenant-a-agent", "tenant-a"),
        _private_record(CapabilityKind.TOOL, "tenant-a-tool", "tenant-a"),
        _private_record(CapabilityKind.AGENT, "tenant-b-agent", "tenant-b"),
        _private_record(CapabilityKind.TOOL, "tenant-b-tool", "tenant-b"),
    )
    source = MarketplaceCapabilityCatalogSource(source=_OFFICIAL, records=records)
    catalog = FederatedCapabilityCatalog((source,))
    return MarketplaceCatalogService(catalog=catalog, marketplace_sources=(source,))


def _global_query() -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
    )


def _tenant_context(tenant_id: str) -> MarketplaceQueryContext:
    return MarketplaceQueryContext(tenant_id=tenant_id)


def _logical_ids(
    views: tuple[object, ...],
) -> set[str]:
    return {view.listing.capability.identity.logical.logical_id for view in views}


def test_tenant_a_sees_public_and_own_private() -> None:
    service = _me9_service()
    views = service.list_listings(
        _global_query(),
        marketplace_query_context=_tenant_context("tenant-a"),
    )
    assert _logical_ids(views) == {
        "public-agent",
        "public-tool",
        "tenant-a-agent",
        "tenant-a-tool",
    }


def test_tenant_b_sees_public_and_own_private() -> None:
    service = _me9_service()
    views = service.list_listings(
        _global_query(),
        marketplace_query_context=_tenant_context("tenant-b"),
    )
    assert _logical_ids(views) == {
        "public-agent",
        "public-tool",
        "tenant-b-agent",
        "tenant-b-tool",
    }


def test_tenant_c_sees_public_only() -> None:
    service = _me9_service()
    views = service.list_listings(
        _global_query(),
        marketplace_query_context=_tenant_context("tenant-c"),
    )
    assert _logical_ids(views) == {"public-agent", "public-tool"}


def test_no_tenant_context_excludes_private() -> None:
    service = _me9_service()
    views = service.list_listings(_global_query())
    assert _logical_ids(views) == {"public-agent", "public-tool"}


def test_foreign_private_lookup_denied() -> None:
    service = _me9_service()
    tenant_a_key = CapabilityIdentityKey(
        kind=CapabilityKind.AGENT,
        source_id=_OFFICIAL.source_id,
        source_kind=_OFFICIAL.source_kind,
        logical_id="tenant-a-agent",
    )
    assert service.get_listing(tenant_a_key, marketplace_query_context=_tenant_context("tenant-b")) is None
    assert service.get_listing(tenant_a_key, marketplace_query_context=_tenant_context("tenant-a")) is not None


def test_search_text_does_not_leak_foreign_private() -> None:
    service = _me9_service()
    views = service.list_listings(
        _global_query(),
        marketplace_query_context=_tenant_context("tenant-b"),
        query_text="tenant-a",
    )
    assert _logical_ids(views) == set()


class _SpyRanker:
    def __init__(self) -> None:
        self.received_logical_ids: list[str] = []

    @property
    def ranker_id(self) -> str:
        return "spy.ranker"

    def rank(self, candidates, context):
        del context
        from intergrax.capability_catalog.ranked_candidate import RankedCapabilityCandidate
        from intergrax.contracts.capability_catalog.ranking import (
            CapabilityRankingEvidence,
            CapabilityRankingSignal,
        )

        ranked: list[RankedCapabilityCandidate] = []
        for index, candidate in enumerate(candidates, start=1):
            self.received_logical_ids.append(candidate.identity.logical.logical_id)
            ranked.append(
                RankedCapabilityCandidate(
                    candidate=candidate,
                    evidence=CapabilityRankingEvidence(
                        ranker_id=self.ranker_id,
                        rank_position=index,
                        signal=CapabilityRankingSignal.STABLE_IDENTITY_ORDER,
                    ),
                ),
            )
        return tuple(ranked)


def test_hidden_private_not_passed_to_ranker() -> None:
    service = _me9_service()
    views = service.list_listings(
        _global_query(),
        marketplace_query_context=_tenant_context("tenant-a"),
    )
    candidates = tuple(
        CapabilityDiscoveryCandidate(
            catalog_entry=view.listing.capability,
            availability=view.availability,
        )
        for view in views
    )
    spy = _SpyRanker()
    rank_capability_candidates(candidates, spy, context=CapabilityRankingContext())
    assert "tenant-b-agent" not in spy.received_logical_ids
    assert "tenant-b-tool" not in spy.received_logical_ids


class _EvilWidenPolicy:
    @property
    def policy_id(self) -> str:
        return "evil.widen"

    def allow(
        self,
        listing_visibility: Visibility,
        query_context: MarketplaceQueryContext,
    ) -> bool:
        del listing_visibility, query_context
        return True


def test_custom_policy_cannot_bypass_hard_tenant_isolation() -> None:
    private = MarketplaceVisibility(
        scope=MarketplaceVisibilityScope.TENANT_PRIVATE,
        tenant_id="tenant-a",
    )
    evaluator = MarketplaceVisibilityEvaluator(extension=_EvilWidenPolicy())
    assert not evaluator.is_visible(
        private,
        MarketplaceQueryContext(tenant_id="tenant-b"),
    )
    assert hard_marketplace_tenant_isolation(
        private,
        MarketplaceQueryContext(tenant_id="tenant-b"),
    ) is False


def test_commercial_metadata_does_not_control_visibility() -> None:
    record = MarketplaceListingRecord(
        kind=CapabilityKind.TOOL,
        logical_id="paid-private-tool",
        publisher="intergrax",
        commercial_metadata=MarketplaceCommercialMetadata(
            commercial_model=CommercialModel.PAID,
            minor_units=100,
            currency_code="USD",
        ),
        visibility=MarketplaceVisibility(
            scope=MarketplaceVisibilityScope.TENANT_PRIVATE,
            tenant_id="tenant-a",
        ),
    )
    source = MarketplaceCapabilityCatalogSource(source=_OFFICIAL, records=(record,))
    catalog = FederatedCapabilityCatalog((source,))
    service = MarketplaceCatalogService(catalog=catalog, marketplace_sources=(source,))
    assert service.list_listings(_global_query()) == ()
    visible = service.list_listings(
        _global_query(),
        marketplace_query_context=_tenant_context("tenant-a"),
    )
    assert len(visible) == 1
    assert visible[0].listing.capability.identity.logical.logical_id == "paid-private-tool"


def test_visibility_round_trip_through_metadata_source() -> None:
    record = _private_record(CapabilityKind.SKILL, "tenant-a-skill", "tenant-a")
    source = MarketplaceCapabilityCatalogSource(source=_OFFICIAL, records=(record,))
    listing = source.read_listings()[0]
    assert listing.visibility is not None
    assert listing.visibility.scope is MarketplaceVisibilityScope.TENANT_PRIVATE


def test_e2e_catalog_to_recommendation_respects_tenant_visibility() -> None:
    service = _me9_service()
    views = service.list_listings(
        _global_query(),
        marketplace_query_context=_tenant_context("tenant-a"),
    )
    candidates = tuple(
        CapabilityDiscoveryCandidate(
            catalog_entry=view.listing.capability,
            availability=view.availability,
        )
        for view in views
    )
    discovery = MarketplaceDiscoveryService.with_defaults()
    ranked = discovery.search_and_rank(candidates)
    governed = govern_capability_candidates(
        ranked,
        evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        context=CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT),
    ).allowed
    recommendations = MarketplaceRecommendationService.with_defaults().recommend(governed)
    recommended_ids = {item.identity.logical.logical_id for item in recommendations}
    assert "tenant-b-agent" not in recommended_ids
    assert "tenant-b-tool" not in recommended_ids
    assert "tenant-a-agent" in recommended_ids

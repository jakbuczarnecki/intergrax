# © Artur Czarnecki. All rights reserved.

"""ME-5-C1 governance-safe recommendation boundary proofs."""

from __future__ import annotations

import pytest

from intergrax.capability_catalog import (
    AvailabilityPreservingGovernanceEvaluator,
    CapabilityCatalogEntry,
    CapabilityDiscoveryCandidate,
    CapabilityRecommendation,
    CapabilityRecommendationError,
    DefaultTopRankedCapabilityRecommendationStrategy,
    GovernedCapabilityCandidate,
    StableIdentityRanker,
    govern_capability_candidates,
    rank_capability_candidates,
    recommend_capability_candidates,
)
from intergrax.contracts.capability_catalog import (
    AvailabilityDisposition,
    CapabilityDiscoveryIdentity,
    CapabilityGovernanceContext,
    CapabilityGovernancePosture,
    CapabilityGovernanceReasonCode,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilityRecommendationContext,
    CapabilityRecommendationEvidence,
    CapabilityRecommendationReasonCode,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
    GovernanceDecisionEvidence,
    GovernanceDisposition,
)

pytestmark = pytest.mark.unit

_OFFICIAL = CapabilitySourceIdentity(
    source_id="official.catalog",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


def _entry(kind: CapabilityKind, logical_id: str) -> CapabilityCatalogEntry:
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=kind,
            source=_OFFICIAL,
            logical=CapabilityLogicalIdentity(kind=kind, logical_id=logical_id),
        ),
        provenance=CapabilityProvenance(source=_OFFICIAL),
        display_label=logical_id,
    )


def _discovery(
    kind: CapabilityKind,
    logical_id: str,
    *,
    availability: AvailabilityDisposition = AvailabilityDisposition.CATALOG_AVAILABLE,
) -> CapabilityDiscoveryCandidate:
    return CapabilityDiscoveryCandidate(
        catalog_entry=_entry(kind, logical_id),
        availability=availability,
    )


def test_governance_denied_top_ranked_candidate_is_not_recommended() -> None:
    candidates = (
        _discovery(
            CapabilityKind.TOOL,
            "tools.rank1",
            availability=AvailabilityDisposition.BLOCKED,
        ),
        _discovery(CapabilityKind.TOOL, "tools.rank2"),
        _discovery(CapabilityKind.TOOL, "tools.rank3"),
    )
    ranked = rank_capability_candidates(candidates, StableIdentityRanker())
    assert ranked[0].candidate.identity.logical.logical_id == "tools.rank1"

    governed_result = govern_capability_candidates(
        ranked,
        evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        context=CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT),
    )
    assert len(governed_result.blocked) == 1
    recommendations = recommend_capability_candidates(
        governed_result.allowed,
        DefaultTopRankedCapabilityRecommendationStrategy(),
        context=CapabilityRecommendationContext(top_n=3),
    )
    logical_ids = tuple(
        item.governed.ranked.candidate.identity.logical.logical_id for item in recommendations
    )
    assert "tools.rank1" not in logical_ids
    assert logical_ids == ("tools.rank2", "tools.rank3")


def test_governance_narrowing_preserves_rank_order_for_recommendation() -> None:
    candidates = (
        _discovery(
            CapabilityKind.TOOL,
            "tools.a",
            availability=AvailabilityDisposition.BLOCKED,
        ),
        _discovery(CapabilityKind.TOOL, "tools.b"),
        _discovery(CapabilityKind.TOOL, "tools.c"),
    )
    ranked = rank_capability_candidates(candidates, StableIdentityRanker())
    governed_result = govern_capability_candidates(
        ranked,
        evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        context=CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT),
    )
    recommendations = recommend_capability_candidates(
        governed_result.allowed,
        DefaultTopRankedCapabilityRecommendationStrategy(),
        context=CapabilityRecommendationContext(top_n=2),
    )
    assert [
        item.ranked.candidate.identity.logical.logical_id for item in recommendations
    ] == ["tools.b", "tools.c"]


def test_custom_recommender_receives_only_governed_candidates() -> None:
    ranked = rank_capability_candidates(
        (_discovery(CapabilityKind.AGENT, "agents.one"),),
        StableIdentityRanker(),
    )
    governed_result = govern_capability_candidates(
        ranked,
        evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        context=CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT),
    )

    received: list[type] = []

    class _CaptureInput:
        @property
        def recommendation_strategy_id(self) -> str:
            return "custom.capture"

        def recommend(self, governed, context):
            del context
            received.extend(type(item) for item in governed)
            item = governed[0]
            return (
                CapabilityRecommendation(
                    governed=item,
                    evidence=CapabilityRecommendationEvidence(
                        recommendation_strategy_id="custom.capture",
                        reason_codes=(CapabilityRecommendationReasonCode.TOP_RANKED,),
                    ),
                ),
            )

    recommend_capability_candidates(governed_result.allowed, _CaptureInput())
    assert received == [GovernedCapabilityCandidate]


def test_recommender_cannot_inject_unknown_governed_candidate() -> None:
    ranked_allowed = rank_capability_candidates(
        (_discovery(CapabilityKind.TOOL, "tools.allowed"),),
        StableIdentityRanker(),
    )
    ranked_unknown = rank_capability_candidates(
        (_discovery(CapabilityKind.TOOL, "tools.unknown"),),
        StableIdentityRanker(),
    )
    governed_result = govern_capability_candidates(
        ranked_allowed,
        evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        context=CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT),
    )
    unknown = GovernedCapabilityCandidate(
        ranked=ranked_unknown[0],
        evidence=(
            GovernanceDecisionEvidence(
                evaluator_id="test.inject",
                disposition=GovernanceDisposition.ALLOWED,
                reason_code=CapabilityGovernanceReasonCode.GOVERNANCE_ALLOWED,
            ),
        ),
    )

    class _InjectUnknown:
        @property
        def recommendation_strategy_id(self) -> str:
            return "inject.unknown"

        def recommend(self, governed, context):
            del context
            return (
                CapabilityRecommendation(
                    governed=unknown,
                    evidence=CapabilityRecommendationEvidence(
                        recommendation_strategy_id="inject.unknown",
                        reason_codes=(CapabilityRecommendationReasonCode.TOP_RANKED,),
                    ),
                ),
            )

    with pytest.raises(CapabilityRecommendationError, match="unknown governed"):
        recommend_capability_candidates(governed_result.allowed, _InjectUnknown())


def test_mixed_agent_tool_skill_governed_recommendation_pipeline() -> None:
    candidates = (
        _discovery(CapabilityKind.AGENT, "agents.alpha"),
        _discovery(CapabilityKind.TOOL, "tools.beta"),
        _discovery(CapabilityKind.SKILL, "skills.gamma"),
    )
    ranked = rank_capability_candidates(candidates, StableIdentityRanker())
    governed_result = govern_capability_candidates(
        ranked,
        evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        context=CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT),
    )
    recommendations = recommend_capability_candidates(
        governed_result.allowed,
        DefaultTopRankedCapabilityRecommendationStrategy(),
        context=CapabilityRecommendationContext(top_n=3),
    )
    kinds = {item.identity.kind for item in recommendations}
    assert kinds == {CapabilityKind.AGENT, CapabilityKind.TOOL, CapabilityKind.SKILL}


def test_end_to_end_search_rank_govern_recommend() -> None:
    from intergrax.marketplace import (
        MarketplaceDiscoveryService,
        MarketplaceRecommendationService,
    )

    candidates = (
        _discovery(CapabilityKind.TOOL, "tools.pipeline.a"),
        _discovery(CapabilityKind.TOOL, "tools.pipeline.b"),
    )
    ranked = MarketplaceDiscoveryService.with_defaults().search_and_rank(candidates)
    governed_result = govern_capability_candidates(
        ranked,
        evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        context=CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT),
    )
    recommendations = MarketplaceRecommendationService.with_defaults().recommend(
        governed_result.allowed,
        recommendation_context=CapabilityRecommendationContext(top_n=2),
    )
    assert len(recommendations) == 2

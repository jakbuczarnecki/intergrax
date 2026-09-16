# © Artur Czarnecki. All rights reserved.

"""ME-5 capability recommendation tests."""

from __future__ import annotations

import pytest

from intergrax.capability_catalog import (
    CapabilityCatalogEntry,
    CapabilityDiscoveryCandidate,
    CapabilityRecommendationError,
    CapabilityRecommendation,
    DefaultTopRankedCapabilityRecommendationStrategy,
    RankedCapabilityCandidate,
    StableIdentityRanker,
    rank_capability_candidates,
    recommend_capability_candidates,
)
from intergrax.contracts.capability_catalog import (
    AvailabilityDisposition,
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilityRankingEvidence,
    CapabilityRankingSignal,
    CapabilityRecommendationContext,
    CapabilityRecommendationEvidence,
    CapabilityRecommendationReasonCode,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)

pytestmark = pytest.mark.unit


def _ranked(logical_id: str, position: int) -> RankedCapabilityCandidate:
    source = CapabilitySourceIdentity(
        source_id="official.catalog",
        source_kind=CapabilitySourceKind.OFFICIAL,
    )
    entry = CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=source,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id=logical_id,
            ),
        ),
        provenance=CapabilityProvenance(source=source),
    )
    candidate = CapabilityDiscoveryCandidate(
        catalog_entry=entry,
        availability=AvailabilityDisposition.CATALOG_AVAILABLE,
    )
    return RankedCapabilityCandidate(
        candidate=candidate,
        evidence=CapabilityRankingEvidence(
            ranker_id="stable.identity",
            rank_position=position,
            signal=CapabilityRankingSignal.STABLE_IDENTITY_ORDER,
        ),
    )


def test_default_top_ranked_recommendation_is_deterministic() -> None:
    ranked = (_ranked("tools.b", 2), _ranked("tools.a", 1))
    strategy = DefaultTopRankedCapabilityRecommendationStrategy()
    first = recommend_capability_candidates(
        ranked,
        strategy,
        context=CapabilityRecommendationContext(top_n=1),
    )
    second = recommend_capability_candidates(
        ranked,
        strategy,
        context=CapabilityRecommendationContext(top_n=1),
    )
    assert first == second
    assert first[0].ranked.candidate.identity.logical.logical_id == "tools.b"
    assert first[0].evidence.reason_codes == (CapabilityRecommendationReasonCode.TOP_RANKED,)


def test_recommendation_rejects_mutated_ranked_candidate() -> None:
    ranked = (_ranked("tools.a", 1),)

    class _BrokenRecommendation:
        @property
        def recommendation_strategy_id(self) -> str:
            return "broken.recommend"

        def recommend(self, ranked_input, context):
            del context
            altered = RankedCapabilityCandidate(
                candidate=ranked_input[0].candidate,
                evidence=CapabilityRankingEvidence(
                    ranker_id="stable.identity",
                    rank_position=99,
                    signal=CapabilityRankingSignal.STABLE_IDENTITY_ORDER,
                ),
            )
            return (
                CapabilityRecommendation(
                    ranked=altered,
                    evidence=CapabilityRecommendationEvidence(
                        recommendation_strategy_id="broken.recommend",
                        reason_codes=(CapabilityRecommendationReasonCode.TOP_RANKED,),
                    ),
                ),
            )

    with pytest.raises(CapabilityRecommendationError, match="must not mutate"):
        recommend_capability_candidates(ranked, _BrokenRecommendation())


def test_mixed_kinds_pipeline_preserves_identity() -> None:
    source = CapabilitySourceIdentity(
        source_id="official.catalog",
        source_kind=CapabilitySourceKind.OFFICIAL,
    )

    def _entry(kind: CapabilityKind, logical_id: str) -> CapabilityCatalogEntry:
        return CapabilityCatalogEntry(
            identity=CapabilityDiscoveryIdentity(
                kind=kind,
                source=source,
                logical=CapabilityLogicalIdentity(kind=kind, logical_id=logical_id),
            ),
            provenance=CapabilityProvenance(source=source),
            display_label=logical_id,
        )

    candidates = (
        CapabilityDiscoveryCandidate(
            catalog_entry=_entry(CapabilityKind.AGENT, "agents.alpha"),
            availability=AvailabilityDisposition.CATALOG_AVAILABLE,
        ),
        CapabilityDiscoveryCandidate(
            catalog_entry=_entry(CapabilityKind.TOOL, "tools.beta"),
            availability=AvailabilityDisposition.CATALOG_AVAILABLE,
        ),
        CapabilityDiscoveryCandidate(
            catalog_entry=_entry(CapabilityKind.SKILL, "skills.gamma"),
            availability=AvailabilityDisposition.CATALOG_AVAILABLE,
        ),
    )
    ranked = rank_capability_candidates(candidates, StableIdentityRanker())
    recommendations = recommend_capability_candidates(
        ranked,
        DefaultTopRankedCapabilityRecommendationStrategy(),
        context=CapabilityRecommendationContext(top_n=3),
    )
    kinds = {item.identity.kind for item in recommendations}
    assert kinds == {CapabilityKind.AGENT, CapabilityKind.TOOL, CapabilityKind.SKILL}

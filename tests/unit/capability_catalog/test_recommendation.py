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
    GovernedCapabilityCandidate,
    RankedCapabilityCandidate,
    StableIdentityRanker,
    rank_capability_candidates,
    recommend_capability_candidates,
)
from intergrax.contracts.capability_catalog import (
    AvailabilityDisposition,
    CapabilityDiscoveryIdentity,
    CapabilityGovernanceReasonCode,
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
    GovernanceDecisionEvidence,
    GovernanceDisposition,
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


def _governed(ranked: RankedCapabilityCandidate) -> GovernedCapabilityCandidate:
    return GovernedCapabilityCandidate(
        ranked=ranked,
        evidence=(
            GovernanceDecisionEvidence(
                evaluator_id="test.fixture",
                disposition=GovernanceDisposition.ALLOWED,
                reason_code=CapabilityGovernanceReasonCode.GOVERNANCE_ALLOWED,
            ),
        ),
    )


def test_default_top_ranked_recommendation_is_deterministic() -> None:
    governed = (
        _governed(_ranked("tools.b", 2)),
        _governed(_ranked("tools.a", 1)),
    )
    strategy = DefaultTopRankedCapabilityRecommendationStrategy()
    first = recommend_capability_candidates(
        governed,
        strategy,
        context=CapabilityRecommendationContext(top_n=1),
    )
    second = recommend_capability_candidates(
        governed,
        strategy,
        context=CapabilityRecommendationContext(top_n=1),
    )
    assert first == second
    assert first[0].ranked.candidate.identity.logical.logical_id == "tools.b"
    assert first[0].evidence.reason_codes == (CapabilityRecommendationReasonCode.TOP_RANKED,)
    assert first[0].governance_evidence[0].disposition is GovernanceDisposition.ALLOWED


def test_recommendation_requires_governed_candidates() -> None:
    ranked = (_ranked("tools.a", 1),)
    with pytest.raises(CapabilityRecommendationError, match="GovernedCapabilityCandidate"):
        recommend_capability_candidates(
            ranked,  # type: ignore[arg-type]
            DefaultTopRankedCapabilityRecommendationStrategy(),
        )


def test_recommendation_rejects_mutated_governed_candidate() -> None:
    governed = (_governed(_ranked("tools.a", 1)),)

    class _BrokenRecommendation:
        @property
        def recommendation_strategy_id(self) -> str:
            return "broken.recommend"

        def recommend(self, governed_input, context):
            del context
            original = governed_input[0]
            altered_ranked = RankedCapabilityCandidate(
                candidate=original.ranked.candidate,
                evidence=CapabilityRankingEvidence(
                    ranker_id="stable.identity",
                    rank_position=99,
                    signal=CapabilityRankingSignal.STABLE_IDENTITY_ORDER,
                ),
            )
            altered = GovernedCapabilityCandidate(
                ranked=altered_ranked,
                evidence=original.evidence,
            )
            return (
                CapabilityRecommendation(
                    governed=altered,
                    evidence=CapabilityRecommendationEvidence(
                        recommendation_strategy_id="broken.recommend",
                        reason_codes=(CapabilityRecommendationReasonCode.TOP_RANKED,),
                    ),
                ),
            )

    with pytest.raises(CapabilityRecommendationError, match="must not mutate"):
        recommend_capability_candidates(governed, _BrokenRecommendation())


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
    governed = tuple(_governed(item) for item in ranked)
    recommendations = recommend_capability_candidates(
        governed,
        DefaultTopRankedCapabilityRecommendationStrategy(),
        context=CapabilityRecommendationContext(top_n=3),
    )
    kinds = {item.identity.kind for item in recommendations}
    assert kinds == {CapabilityKind.AGENT, CapabilityKind.TOOL, CapabilityKind.SKILL}

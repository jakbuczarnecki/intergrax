# © Artur Czarnecki. All rights reserved.

"""ME-6 — governance / trust / availability narrowing qualification proofs."""

from __future__ import annotations

import pytest

from intergrax.capability_catalog import (
    AvailabilityPreservingGovernanceEvaluator,
    CapabilityCatalogEntry,
    CapabilityDiscoveryCandidate,
    CapabilityGovernanceError,
    DefaultTopRankedCapabilityRecommendationStrategy,
    GovernedCapabilityCandidate,
    RankedCapabilityCandidate,
    StableIdentityRanker,
    govern_capability_candidates,
    rank_capability_candidates,
    recommend_capability_candidates,
)
from intergrax.capability_catalog.adapters.agent_governance import AgentTrustGovernanceEvaluator
from intergrax.capability_catalog.adapters.skill_governance import SkillProfileGovernanceEvaluator
from intergrax.capability_catalog.adapters.tool_governance import ToolPolicyGovernanceEvaluator
from intergrax.capability_catalog.governance import CapabilityGovernanceDecision
from intergrax.marketplace.recommendation import MarketplaceRecommendationService
from intergrax.contracts.capability_catalog import (
    AvailabilityDisposition,
    CapabilityAgentGovernanceEvidence,
    CapabilityDiscoveryIdentity,
    CapabilityGovernanceContext,
    CapabilityGovernancePosture,
    CapabilityGovernanceReasonCode,
    CapabilityIdentityKey,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilityRankingEvidence,
    CapabilityRankingSignal,
    CapabilityRecommendationContext,
    CapabilitySetConstraintMode,
    CapabilitySkillGovernanceEvidence,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
    CapabilityToolGovernanceEvidence,
    GovernanceDecisionEvidence,
    GovernanceDisposition,
)

pytestmark = pytest.mark.unit

_OFFICIAL = CapabilitySourceIdentity(
    source_id="official.catalog",
    source_kind=CapabilitySourceKind.OFFICIAL,
)

_BASELINE = AvailabilityPreservingGovernanceEvaluator()
_POLICY = ToolPolicyGovernanceEvaluator()
_TRUST = AgentTrustGovernanceEvaluator()
_SKILL = SkillProfileGovernanceEvaluator()
_PRODUCTION_EVALUATORS = (_BASELINE, _POLICY, _TRUST, _SKILL)


def _entry(kind: CapabilityKind, logical_id: str) -> CapabilityCatalogEntry:
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=kind,
            source=_OFFICIAL,
            logical=CapabilityLogicalIdentity(kind=kind, logical_id=logical_id),
        ),
        provenance=CapabilityProvenance(source=_OFFICIAL, version_label="1.0.0"),
        display_label=logical_id,
    )


def _ranked(
    kind: CapabilityKind,
    logical_id: str,
    *,
    availability: AvailabilityDisposition = AvailabilityDisposition.HOST_AVAILABLE,
    position: int = 1,
) -> RankedCapabilityCandidate:
    return RankedCapabilityCandidate(
        candidate=CapabilityDiscoveryCandidate(
            catalog_entry=_entry(kind, logical_id),
            availability=availability,
        ),
        evidence=CapabilityRankingEvidence(
            ranker_id="stable.identity",
            rank_position=position,
            signal=CapabilityRankingSignal.STABLE_IDENTITY_ORDER,
            original_stage3_position=position,
        ),
    )


def _strict_context(
    *,
    tool: CapabilityToolGovernanceEvidence | None = None,
    agent: CapabilityAgentGovernanceEvidence | None = None,
    skill: CapabilitySkillGovernanceEvidence | None = None,
) -> CapabilityGovernanceContext:
    return CapabilityGovernanceContext(
        posture=CapabilityGovernancePosture.STRICT,
        tool_evidence=tool,
        agent_evidence=agent,
        skill_evidence=skill,
    )


def test_all_required_governance_signals_allow_candidate() -> None:
    tool = _ranked(CapabilityKind.TOOL, "tools.alpha")
    agent = _ranked(CapabilityKind.AGENT, "agents.beta")
    skill = _ranked(CapabilityKind.SKILL, "skills.gamma")
    tool_key = CapabilityIdentityKey.from_discovery_identity(tool.identity)
    agent_key = CapabilityIdentityKey.from_discovery_identity(agent.identity)
    skill_key = CapabilityIdentityKey.from_discovery_identity(skill.identity)
    context = _strict_context(
        tool=CapabilityToolGovernanceEvidence(
            allowed_keys=(tool_key,),
            allowed_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
        ),
        agent=CapabilityAgentGovernanceEvidence(trusted_keys=(agent_key,)),
        skill=CapabilitySkillGovernanceEvidence(
            enabled_keys=(skill_key,),
            enabled_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
        ),
    )
    result = govern_capability_candidates(
        (tool, agent, skill),
        evaluators=_PRODUCTION_EVALUATORS,
        context=context,
    )
    assert len(result.allowed) == 3
    assert not result.blocked
    for item in result.allowed:
        assert isinstance(item, GovernedCapabilityCandidate)


def test_trust_block_prevents_admission() -> None:
    agent = _ranked(CapabilityKind.AGENT, "agents.revoked")
    agent_key = CapabilityIdentityKey.from_discovery_identity(agent.identity)
    context = _strict_context(
        agent=CapabilityAgentGovernanceEvidence(revoked_keys=(agent_key,)),
    )
    result = govern_capability_candidates(
        (agent,),
        evaluators=_PRODUCTION_EVALUATORS,
        context=context,
    )
    assert not result.allowed
    assert any(
        item.reason_code is CapabilityGovernanceReasonCode.TRUST_NOT_SATISFIED
        for item in result.blocked[0].evidence
    )


def test_availability_block_prevents_admission() -> None:
    tool = _ranked(
        CapabilityKind.TOOL,
        "tools.down",
        availability=AvailabilityDisposition.UNAVAILABLE,
    )
    tool_key = CapabilityIdentityKey.from_discovery_identity(tool.identity)
    context = _strict_context(
        tool=CapabilityToolGovernanceEvidence(
            allowed_keys=(tool_key,),
            allowed_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
        ),
    )
    result = govern_capability_candidates(
        (tool,),
        evaluators=_PRODUCTION_EVALUATORS,
        context=context,
    )
    assert not result.allowed
    assert any(
        item.reason_code is CapabilityGovernanceReasonCode.AVAILABILITY_UNAVAILABLE
        for item in result.blocked[0].evidence
    )


def test_policy_block_prevents_admission() -> None:
    tool = _ranked(CapabilityKind.TOOL, "tools.denied")
    tool_key = CapabilityIdentityKey.from_discovery_identity(tool.identity)
    context = _strict_context(
        tool=CapabilityToolGovernanceEvidence(denied_keys=(tool_key,)),
    )
    result = govern_capability_candidates(
        (tool,),
        evaluators=_PRODUCTION_EVALUATORS,
        context=context,
    )
    assert not result.allowed
    assert any(
        item.reason_code is CapabilityGovernanceReasonCode.POLICY_DENIED
        for item in result.blocked[0].evidence
    )


def test_missing_mandatory_evidence_fails_closed() -> None:
    agent = _ranked(CapabilityKind.AGENT, "agents.no_evidence")
    context = CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT)
    result = govern_capability_candidates(
        (agent,),
        evaluators=_PRODUCTION_EVALUATORS,
        context=context,
    )
    assert not result.allowed
    assert any(
        item.reason_code is CapabilityGovernanceReasonCode.MISSING_REQUIRED_EVIDENCE
        for item in result.blocked[0].evidence
    )


def test_governance_preserves_rank_order() -> None:
    ranked = (
        _ranked(CapabilityKind.TOOL, "tools.a", position=1),
        _ranked(
            CapabilityKind.TOOL,
            "tools.b",
            availability=AvailabilityDisposition.BLOCKED,
            position=2,
        ),
        _ranked(CapabilityKind.TOOL, "tools.c", position=3),
        _ranked(
            CapabilityKind.TOOL,
            "tools.d",
            availability=AvailabilityDisposition.SCOPE_UNAVAILABLE,
            position=4,
        ),
    )
    result = govern_capability_candidates(
        ranked,
        evaluators=(_BASELINE,),
        context=_strict_context(),
    )
    assert [item.identity.logical.logical_id for item in result.allowed] == [
        "tools.a",
        "tools.c",
    ]
    assert [item.identity.logical.logical_id for item in result.blocked] == [
        "tools.b",
        "tools.d",
    ]


def test_mixed_agent_tool_skill_governance_pipeline() -> None:
    tool = _ranked(CapabilityKind.TOOL, "tools.one", position=1)
    agent = _ranked(CapabilityKind.AGENT, "agents.two", position=2)
    skill = _ranked(CapabilityKind.SKILL, "skills.three", position=3)
    tool_key = CapabilityIdentityKey.from_discovery_identity(tool.identity)
    agent_key = CapabilityIdentityKey.from_discovery_identity(agent.identity)
    skill_key = CapabilityIdentityKey.from_discovery_identity(skill.identity)
    context = _strict_context(
        tool=CapabilityToolGovernanceEvidence(
            allowed_keys=(tool_key,),
            allowed_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
        ),
        agent=CapabilityAgentGovernanceEvidence(trusted_keys=(agent_key,)),
        skill=CapabilitySkillGovernanceEvidence(
            enabled_keys=(skill_key,),
            enabled_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
        ),
    )
    result = govern_capability_candidates(
        (tool, agent, skill),
        evaluators=_PRODUCTION_EVALUATORS,
        context=context,
    )
    assert [item.identity.logical.logical_id for item in result.allowed] == [
        "tools.one",
        "agents.two",
        "skills.three",
    ]


def test_blocked_candidate_never_reaches_recommendation() -> None:
    candidates = (
        CapabilityDiscoveryCandidate(
            catalog_entry=_entry(CapabilityKind.TOOL, "tools.rank1"),
            availability=AvailabilityDisposition.BLOCKED,
        ),
        CapabilityDiscoveryCandidate(
            catalog_entry=_entry(CapabilityKind.TOOL, "tools.rank2"),
            availability=AvailabilityDisposition.CATALOG_AVAILABLE,
        ),
    )
    ranked = rank_capability_candidates(candidates, StableIdentityRanker())
    governed = govern_capability_candidates(
        ranked,
        evaluators=(_BASELINE,),
        context=_strict_context(),
    )
    service = MarketplaceRecommendationService.with_defaults()
    recommendations = service.recommend(
        governed.allowed,
        recommendation_context=CapabilityRecommendationContext(top_n=5),
    )
    logical_ids = tuple(
        item.governed.ranked.candidate.identity.logical.logical_id
        for item in recommendations
    )
    assert "tools.rank1" not in logical_ids
    assert logical_ids == ("tools.rank2",)


class _CustomBlockEvaluator:
    @property
    def evaluator_id(self) -> str:
        return "custom.me6.block"

    def evaluate(
        self,
        candidate: RankedCapabilityCandidate,
        context: CapabilityGovernanceContext,
    ) -> CapabilityGovernanceDecision:
        del context
        if candidate.identity.logical.logical_id == "tools.blocked":
            return CapabilityGovernanceDecision(
                disposition=GovernanceDisposition.BLOCKED,
                evidence=GovernanceDecisionEvidence(
                    evaluator_id=self.evaluator_id,
                    disposition=GovernanceDisposition.BLOCKED,
                    reason_code=CapabilityGovernanceReasonCode.POLICY_DENIED,
                    detail="custom plugin block",
                ),
            )
        return CapabilityGovernanceDecision(
            disposition=GovernanceDisposition.ALLOWED,
            evidence=GovernanceDecisionEvidence(
                evaluator_id=self.evaluator_id,
                disposition=GovernanceDisposition.ALLOWED,
                reason_code=CapabilityGovernanceReasonCode.GOVERNANCE_ALLOWED,
            ),
        )


def test_custom_governance_evaluator_plugs_in_without_core_changes() -> None:
    ranked = (
        _ranked(CapabilityKind.TOOL, "tools.ok"),
        _ranked(CapabilityKind.TOOL, "tools.blocked"),
    )
    result = govern_capability_candidates(
        ranked,
        evaluators=(_CustomBlockEvaluator(),),
        context=_strict_context(),
    )
    assert len(result.allowed) == 1
    assert result.allowed[0].identity.logical.logical_id == "tools.ok"
    assert result.blocked[0].evidence[0].evaluator_id == "custom.me6.block"


class _CustomTrustProjectionEvaluator:
    """Plugin trust evidence — structural CapabilityGovernanceEvaluator, not a core subclass."""

    @property
    def evaluator_id(self) -> str:
        return "custom.trust.projection"

    def evaluate(
        self,
        candidate: RankedCapabilityCandidate,
        context: CapabilityGovernanceContext,
    ) -> CapabilityGovernanceDecision:
        del context
        if candidate.identity.kind is not CapabilityKind.AGENT:
            return CapabilityGovernanceDecision(
                disposition=GovernanceDisposition.ALLOWED,
                evidence=GovernanceDecisionEvidence(
                    evaluator_id=self.evaluator_id,
                    disposition=GovernanceDisposition.ALLOWED,
                    reason_code=CapabilityGovernanceReasonCode.GOVERNANCE_NOT_APPLICABLE,
                ),
            )
        return CapabilityGovernanceDecision(
            disposition=GovernanceDisposition.BLOCKED,
            evidence=GovernanceDecisionEvidence(
                evaluator_id=self.evaluator_id,
                disposition=GovernanceDisposition.BLOCKED,
                reason_code=CapabilityGovernanceReasonCode.TRUST_NOT_SATISFIED,
                reference="trust://custom/projection",
            ),
        )


def test_custom_trust_evidence_provider_plugs_into_governance() -> None:
    agent = _ranked(CapabilityKind.AGENT, "agents.external")
    result = govern_capability_candidates(
        (agent,),
        evaluators=(_BASELINE, _CustomTrustProjectionEvaluator()),
        context=_strict_context(),
    )
    assert not result.allowed
    assert result.blocked[0].evidence[-1].reference == "trust://custom/projection"


class _CustomAvailabilityProjectionEvaluator:
    @property
    def evaluator_id(self) -> str:
        return "custom.availability.projection"

    def evaluate(
        self,
        candidate: RankedCapabilityCandidate,
        context: CapabilityGovernanceContext,
    ) -> CapabilityGovernanceDecision:
        del candidate, context
        return CapabilityGovernanceDecision(
            disposition=GovernanceDisposition.BLOCKED,
            evidence=GovernanceDecisionEvidence(
                evaluator_id=self.evaluator_id,
                disposition=GovernanceDisposition.BLOCKED,
                reason_code=CapabilityGovernanceReasonCode.AVAILABILITY_SCOPE_UNAVAILABLE,
                detail="custom availability projection",
            ),
        )


def test_custom_availability_evidence_provider_plugs_into_governance() -> None:
    tool = _ranked(CapabilityKind.TOOL, "tools.host")
    result = govern_capability_candidates(
        (tool,),
        evaluators=(_BASELINE, _CustomAvailabilityProjectionEvaluator()),
        context=_strict_context(),
    )
    assert not result.allowed


class _TypedFailureEvaluator:
    @property
    def evaluator_id(self) -> str:
        return "custom.typed_failure"

    def evaluate(
        self,
        candidate: RankedCapabilityCandidate,
        context: CapabilityGovernanceContext,
    ) -> CapabilityGovernanceDecision:
        del candidate, context
        raise CapabilityGovernanceError("expected governance failure")


def test_typed_provider_failure_surfaces_as_governance_error() -> None:
    ranked = (_ranked(CapabilityKind.TOOL, "tools.one"),)
    with pytest.raises(CapabilityGovernanceError, match="expected governance failure"):
        govern_capability_candidates(
            ranked,
            evaluators=(_TypedFailureEvaluator(),),
            context=_strict_context(),
        )


class _UnexpectedFailureEvaluator:
    @property
    def evaluator_id(self) -> str:
        return "custom.unexpected"

    def evaluate(
        self,
        candidate: RankedCapabilityCandidate,
        context: CapabilityGovernanceContext,
    ) -> CapabilityGovernanceDecision:
        del candidate, context
        raise RuntimeError("programming defect")


def test_unexpected_evaluator_failure_propagates_in_non_strict_posture() -> None:
    ranked = (_ranked(CapabilityKind.TOOL, "tools.one"),)
    context = CapabilityGovernanceContext(posture=CapabilityGovernancePosture.NON_STRICT)
    with pytest.raises(CapabilityGovernanceError, match="programming defect"):
        govern_capability_candidates(
            ranked,
            evaluators=(_UnexpectedFailureEvaluator(),),
            context=context,
        )


def test_ranked_governed_recommendation_integration_chain() -> None:
    candidates = (
        CapabilityDiscoveryCandidate(
            catalog_entry=_entry(CapabilityKind.TOOL, "tools.first"),
            availability=AvailabilityDisposition.CATALOG_AVAILABLE,
        ),
        CapabilityDiscoveryCandidate(
            catalog_entry=_entry(CapabilityKind.TOOL, "tools.second"),
            availability=AvailabilityDisposition.BLOCKED,
        ),
    )
    ranked = rank_capability_candidates(candidates, StableIdentityRanker())
    governed = govern_capability_candidates(
        ranked,
        evaluators=(_BASELINE,),
        context=_strict_context(),
    )
    recommendations = recommend_capability_candidates(
        governed.allowed,
        DefaultTopRankedCapabilityRecommendationStrategy(),
        context=CapabilityRecommendationContext(top_n=3),
    )
    assert [item.governed.identity.logical.logical_id for item in recommendations] == [
        "tools.first",
    ]

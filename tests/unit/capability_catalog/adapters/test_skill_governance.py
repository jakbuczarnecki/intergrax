# © Artur Czarnecki. All rights reserved.

"""Skill profile governance adapter tests (Stage 5)."""

from __future__ import annotations

import pytest

from intergrax.capability_catalog.adapters.skill_governance import (
    SKILL_PROFILE_GOVERNANCE_EVALUATOR_ID,
    SkillProfileGovernanceEvaluator,
)
from intergrax.capability_catalog import (
    AvailabilityPreservingGovernanceEvaluator,
    CapabilityCatalogEntry,
    CapabilityDiscoveryCandidate,
    RankedCapabilityCandidate,
    govern_capability_candidates,
)
from intergrax.contracts.capability_catalog import (
    AvailabilityDisposition,
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
    CapabilitySkillGovernanceEvidence,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)

pytestmark = pytest.mark.unit


def _skill_ranked(logical_id: str = "skills.echo") -> RankedCapabilityCandidate:
    source = CapabilitySourceIdentity(
        source_id="skills.catalog.builtin",
        source_kind=CapabilitySourceKind.BUILTIN,
    )
    entry = CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.SKILL,
            source=source,
            logical=CapabilityLogicalIdentity(kind=CapabilityKind.SKILL, logical_id=logical_id),
        ),
        provenance=CapabilityProvenance(source=source),
        display_label=logical_id,
    )
    return RankedCapabilityCandidate(
        candidate=CapabilityDiscoveryCandidate(
            catalog_entry=entry,
            availability=AvailabilityDisposition.CATALOG_AVAILABLE,
        ),
        evidence=CapabilityRankingEvidence(
            ranker_id="stable.identity",
            rank_position=1,
            signal=CapabilityRankingSignal.STABLE_IDENTITY_ORDER,
        ),
    )


def test_enabled_skill_allowed() -> None:
    ranked = _skill_ranked()
    key = CapabilityIdentityKey.from_discovery_identity(ranked.identity)
    context = CapabilityGovernanceContext(
        posture=CapabilityGovernancePosture.STRICT,
        skill_evidence=CapabilitySkillGovernanceEvidence(enabled_keys=(key,)),
    )
    result = govern_capability_candidates(
        (ranked,),
        evaluators=(
            AvailabilityPreservingGovernanceEvaluator(),
            SkillProfileGovernanceEvaluator(),
        ),
        context=context,
    )
    assert len(result.allowed) == 1


def test_blocked_skill_evidence_blocks() -> None:
    ranked = _skill_ranked()
    key = CapabilityIdentityKey.from_discovery_identity(ranked.identity)
    context = CapabilityGovernanceContext(
        posture=CapabilityGovernancePosture.STRICT,
        skill_evidence=CapabilitySkillGovernanceEvidence(blocked_keys=(key,)),
    )
    result = govern_capability_candidates(
        (ranked,),
        evaluators=(
            AvailabilityPreservingGovernanceEvaluator(),
            SkillProfileGovernanceEvaluator(),
        ),
        context=context,
    )
    assert not result.allowed
    assert any(
        item.evaluator_id == SKILL_PROFILE_GOVERNANCE_EVALUATOR_ID
        and item.reason_code is CapabilityGovernanceReasonCode.POLICY_DENIED
        for item in result.blocked[0].evidence
    )


def test_missing_skill_evidence_strict_blocks() -> None:
    ranked = _skill_ranked()
    context = CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT)
    result = govern_capability_candidates(
        (ranked,),
        evaluators=(
            AvailabilityPreservingGovernanceEvaluator(),
            SkillProfileGovernanceEvaluator(),
        ),
        context=context,
    )
    assert not result.allowed

# © Artur Czarnecki. All rights reserved.

"""Agent trust governance adapter tests (Stage 5)."""

from __future__ import annotations

import pytest

from intergrax.capability_catalog.adapters.agent_governance import (
    AGENT_TRUST_GOVERNANCE_EVALUATOR_ID,
    AgentTrustGovernanceEvaluator,
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
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)

pytestmark = pytest.mark.unit


def _agent_ranked(logical_id: str = "agents.alpha") -> RankedCapabilityCandidate:
    source = CapabilitySourceIdentity(
        source_id="official.catalog",
        source_kind=CapabilitySourceKind.OFFICIAL,
    )
    entry = CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.AGENT,
            source=source,
            logical=CapabilityLogicalIdentity(kind=CapabilityKind.AGENT, logical_id=logical_id),
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


def test_trusted_agent_allowed() -> None:
    ranked = _agent_ranked()
    key = CapabilityIdentityKey.from_discovery_identity(ranked.identity)
    context = CapabilityGovernanceContext(
        posture=CapabilityGovernancePosture.STRICT,
        agent_evidence=CapabilityAgentGovernanceEvidence(trusted_keys=(key,)),
    )
    result = govern_capability_candidates(
        (ranked,),
        evaluators=(
            AvailabilityPreservingGovernanceEvaluator(),
            AgentTrustGovernanceEvaluator(),
        ),
        context=context,
    )
    assert len(result.allowed) == 1


def test_revoked_agent_blocked() -> None:
    ranked = _agent_ranked()
    key = CapabilityIdentityKey.from_discovery_identity(ranked.identity)
    context = CapabilityGovernanceContext(
        posture=CapabilityGovernancePosture.STRICT,
        agent_evidence=CapabilityAgentGovernanceEvidence(revoked_keys=(key,)),
    )
    result = govern_capability_candidates(
        (ranked,),
        evaluators=(
            AvailabilityPreservingGovernanceEvaluator(),
            AgentTrustGovernanceEvaluator(),
        ),
        context=context,
    )
    assert not result.allowed
    assert any(
        item.evaluator_id == AGENT_TRUST_GOVERNANCE_EVALUATOR_ID
        and item.reason_code is CapabilityGovernanceReasonCode.TRUST_NOT_SATISFIED
        for item in result.blocked[0].evidence
    )


def test_missing_agent_evidence_strict_blocks() -> None:
    ranked = _agent_ranked()
    context = CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT)
    result = govern_capability_candidates(
        (ranked,),
        evaluators=(
            AvailabilityPreservingGovernanceEvaluator(),
            AgentTrustGovernanceEvaluator(),
        ),
        context=context,
    )
    assert not result.allowed

# © Artur Czarnecki. All rights reserved.

"""Tool governance adapter tests (Stage 5)."""

from __future__ import annotations

import pytest

from intergrax.capability_catalog.adapters.tool_governance import (
    TOOL_POLICY_GOVERNANCE_EVALUATOR_ID,
    ToolPolicyGovernanceEvaluator,
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
    CapabilitySetConstraintMode,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
    CapabilityToolGovernanceEvidence,
    GovernanceDisposition,
)

pytestmark = pytest.mark.unit


def _tool_ranked(logical_id: str = "tools.echo.ping") -> RankedCapabilityCandidate:
    source = CapabilitySourceIdentity(
        source_id="tools.catalog.builtin",
        source_kind=CapabilitySourceKind.BUILTIN,
    )
    entry = CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=source,
            logical=CapabilityLogicalIdentity(kind=CapabilityKind.TOOL, logical_id=logical_id),
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


def _identity_key(ranked: RankedCapabilityCandidate) -> CapabilityIdentityKey:
    return CapabilityIdentityKey.from_discovery_identity(ranked.identity)


def test_tool_policy_deny_blocks_candidate() -> None:
    ranked = _tool_ranked()
    key = _identity_key(ranked)
    context = CapabilityGovernanceContext(
        posture=CapabilityGovernancePosture.STRICT,
        tool_evidence=CapabilityToolGovernanceEvidence(denied_keys=(key,)),
    )
    result = govern_capability_candidates(
        (ranked,),
        evaluators=(
            AvailabilityPreservingGovernanceEvaluator(),
            ToolPolicyGovernanceEvaluator(),
        ),
        context=context,
    )
    assert not result.allowed
    assert result.blocked[0].evidence[-1].reason_code is (
        CapabilityGovernanceReasonCode.POLICY_DENIED
    )


def test_tool_policy_allow_passes() -> None:
    ranked = _tool_ranked()
    key = _identity_key(ranked)
    context = CapabilityGovernanceContext(
        posture=CapabilityGovernancePosture.STRICT,
        tool_evidence=CapabilityToolGovernanceEvidence(
            allowed_keys=(key,),
            allowed_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
        ),
    )
    result = govern_capability_candidates(
        (ranked,),
        evaluators=(
            AvailabilityPreservingGovernanceEvaluator(),
            ToolPolicyGovernanceEvaluator(),
        ),
        context=context,
    )
    assert len(result.allowed) == 1
    assert not result.blocked


def test_missing_tool_evidence_strict_blocks() -> None:
    ranked = _tool_ranked()
    context = CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT)
    result = govern_capability_candidates(
        (ranked,),
        evaluators=(
            AvailabilityPreservingGovernanceEvaluator(),
            ToolPolicyGovernanceEvaluator(),
        ),
        context=context,
    )
    assert not result.allowed
    assert any(
        item.evaluator_id == TOOL_POLICY_GOVERNANCE_EVALUATOR_ID
        and item.reason_code is CapabilityGovernanceReasonCode.MISSING_REQUIRED_EVIDENCE
        for item in result.blocked[0].evidence
    )


def test_evaluator_does_not_execute_tools() -> None:
    evaluator = ToolPolicyGovernanceEvaluator()
    ranked = _tool_ranked()
    decision = evaluator.evaluate(
        ranked,
        CapabilityGovernanceContext(
            posture=CapabilityGovernancePosture.NON_STRICT,
            tool_evidence=CapabilityToolGovernanceEvidence(),
        ),
    )
    assert decision.disposition is GovernanceDisposition.ALLOWED


def test_explicit_empty_allowed_set_blocks_tool_candidate() -> None:
    ranked = _tool_ranked()
    context = CapabilityGovernanceContext(
        posture=CapabilityGovernancePosture.STRICT,
        tool_evidence=CapabilityToolGovernanceEvidence(
            allowed_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
        ),
    )
    result = govern_capability_candidates(
        (ranked,),
        evaluators=(
            AvailabilityPreservingGovernanceEvaluator(),
            ToolPolicyGovernanceEvaluator(),
        ),
        context=context,
    )
    assert not result.allowed
    assert result.blocked[0].evidence[-1].reason_code is (
        CapabilityGovernanceReasonCode.NOT_ENTITLED
    )


def test_explicit_non_empty_allowed_set_blocks_non_member() -> None:
    ranked = _tool_ranked("tools.echo.other")
    allowed_ranked = _tool_ranked("tools.echo.ping")
    allowed_key = _identity_key(allowed_ranked)
    context = CapabilityGovernanceContext(
        posture=CapabilityGovernancePosture.STRICT,
        tool_evidence=CapabilityToolGovernanceEvidence(
            allowed_keys=(allowed_key,),
            allowed_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
        ),
    )
    result = govern_capability_candidates(
        (ranked, allowed_ranked),
        evaluators=(
            AvailabilityPreservingGovernanceEvaluator(),
            ToolPolicyGovernanceEvaluator(),
        ),
        context=context,
    )
    assert len(result.allowed) == 1
    assert result.allowed[0].identity.logical.logical_id == "tools.echo.ping"
    assert len(result.blocked) == 1
    assert result.blocked[0].identity.logical.logical_id == "tools.echo.other"
    assert result.blocked[0].evidence[-1].reason_code is (
        CapabilityGovernanceReasonCode.NOT_ENTITLED
    )


def test_unconstrained_empty_allowed_set_does_not_narrow() -> None:
    ranked = _tool_ranked()
    context = CapabilityGovernanceContext(
        posture=CapabilityGovernancePosture.NON_STRICT,
        tool_evidence=CapabilityToolGovernanceEvidence(),
    )
    result = govern_capability_candidates(
        (ranked,),
        evaluators=(
            AvailabilityPreservingGovernanceEvaluator(),
            ToolPolicyGovernanceEvaluator(),
        ),
        context=context,
    )
    assert len(result.allowed) == 1

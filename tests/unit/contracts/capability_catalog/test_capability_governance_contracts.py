# © Artur Czarnecki. All rights reserved.

"""CAPABILITY-CATALOG-1 Stage 5 governance contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.contracts.capability_catalog import (
    CapabilityAgentGovernanceEvidence,
    CapabilityGovernanceContext,
    CapabilityGovernancePosture,
    CapabilityGovernanceReasonCode,
    CapabilityIdentityKey,
    CapabilityKind,
    CapabilitySkillGovernanceEvidence,
    CapabilitySourceKind,
    CapabilityToolGovernanceEvidence,
    GovernanceDecisionEvidence,
    GovernanceDisposition,
)

pytestmark = pytest.mark.unit


def _key(logical_id: str, kind: CapabilityKind = CapabilityKind.TOOL) -> CapabilityIdentityKey:
    return CapabilityIdentityKey(
        kind=kind,
        source_id="official.catalog",
        source_kind=CapabilitySourceKind.OFFICIAL,
        logical_id=logical_id,
    )


def test_governance_decision_evidence_requires_evaluator_id() -> None:
    with pytest.raises(ValidationError):
        GovernanceDecisionEvidence(
            evaluator_id="",
            disposition=GovernanceDisposition.ALLOWED,
            reason_code=CapabilityGovernanceReasonCode.GOVERNANCE_ALLOWED,
        )


def test_tool_evidence_rejects_conflicting_keys() -> None:
    key = _key("tools.echo")
    with pytest.raises(ValidationError):
        CapabilityToolGovernanceEvidence(
            allowed_keys=(key,),
            denied_keys=(key,),
        )


def test_agent_evidence_rejects_conflicting_keys() -> None:
    key = _key("agents.alpha", CapabilityKind.AGENT)
    with pytest.raises(ValidationError):
        CapabilityAgentGovernanceEvidence(
            trusted_keys=(key,),
            blocked_keys=(key,),
        )


def test_skill_evidence_rejects_conflicting_keys() -> None:
    key = _key("skills.echo", CapabilityKind.SKILL)
    with pytest.raises(ValidationError):
        CapabilitySkillGovernanceEvidence(
            enabled_keys=(key,),
            blocked_keys=(key,),
        )


def test_governance_context_defaults_to_strict() -> None:
    context = CapabilityGovernanceContext()
    assert context.posture is CapabilityGovernancePosture.STRICT

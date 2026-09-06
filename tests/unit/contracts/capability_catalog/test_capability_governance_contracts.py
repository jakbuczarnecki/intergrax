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
    CapabilitySetConstraintMode,
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
            allowed_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
        )


def test_tool_evidence_unconstrained_empty_allowed_keys_valid() -> None:
    evidence = CapabilityToolGovernanceEvidence()
    assert evidence.allowed_constraint_mode is CapabilitySetConstraintMode.UNCONSTRAINED
    assert evidence.allowed_keys == ()


def test_tool_evidence_unconstrained_non_empty_allowed_keys_invalid() -> None:
    key = _key("tools.echo")
    with pytest.raises(ValidationError, match="allowed_keys must be empty"):
        CapabilityToolGovernanceEvidence(allowed_keys=(key,))


def test_tool_evidence_explicit_set_empty_allowed_keys_valid() -> None:
    evidence = CapabilityToolGovernanceEvidence(
        allowed_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
    )
    assert evidence.allowed_keys == ()


def test_tool_evidence_explicit_set_non_empty_allowed_keys_valid() -> None:
    key = _key("tools.echo")
    evidence = CapabilityToolGovernanceEvidence(
        allowed_keys=(key,),
        allowed_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
    )
    assert evidence.allowed_keys == (key,)


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
            enabled_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
        )


def test_skill_evidence_unconstrained_empty_enabled_keys_valid() -> None:
    evidence = CapabilitySkillGovernanceEvidence()
    assert evidence.enabled_constraint_mode is CapabilitySetConstraintMode.UNCONSTRAINED
    assert evidence.enabled_keys == ()


def test_skill_evidence_unconstrained_non_empty_enabled_keys_invalid() -> None:
    key = _key("skills.echo", CapabilityKind.SKILL)
    with pytest.raises(ValidationError, match="enabled_keys must be empty"):
        CapabilitySkillGovernanceEvidence(enabled_keys=(key,))


def test_skill_evidence_explicit_set_empty_enabled_keys_valid() -> None:
    evidence = CapabilitySkillGovernanceEvidence(
        enabled_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
    )
    assert evidence.enabled_keys == ()


def test_skill_evidence_explicit_set_non_empty_enabled_keys_valid() -> None:
    key = _key("skills.echo", CapabilityKind.SKILL)
    evidence = CapabilitySkillGovernanceEvidence(
        enabled_keys=(key,),
        enabled_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
    )
    assert evidence.enabled_keys == (key,)


def test_governance_context_defaults_to_strict() -> None:
    context = CapabilityGovernanceContext()
    assert context.posture is CapabilityGovernancePosture.STRICT

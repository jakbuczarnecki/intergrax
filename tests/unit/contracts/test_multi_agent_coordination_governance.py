# © Artur Czarnecki. All rights reserved.

"""NPSC-5D/R1 — multi-agent coordination governance contract tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.control_plane_mutation import GovernanceEvaluationPoint
from intergrax.contracts.multi_agent_coordination_governance import (
    MultiAgentCoordinationCapabilityKind,
    MultiAgentCoordinationExecutionMode,
    MultiAgentCoordinationGovernanceContribution,
    MultiAgentCoordinationGovernancePolicyRule,
    MultiAgentCoordinationGovernanceRequest,
    evidence_from_request_and_decision,
    multi_agent_coordination_governance_request_digest,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

pytestmark = pytest.mark.unit


def _principal() -> RequestIdentity:
    return RequestIdentity(
        tenant_id="tenant-a",
        user_id="user-1",
        auth_subject="subject-1",
    )


def _request(
    *,
    mode: MultiAgentCoordinationExecutionMode = MultiAgentCoordinationExecutionMode.SINGLE,
) -> MultiAgentCoordinationGovernanceRequest:
    return MultiAgentCoordinationGovernanceRequest(
        intent_id="intent-1",
        execution_mode=mode,
        contributions=(
            MultiAgentCoordinationGovernanceContribution(
                contribution_id="contrib-a",
                capability_kind=MultiAgentCoordinationCapabilityKind.RESOLVED_REQUIREMENT,
                required_capability_ids=("document.ocr",),
            ),
        ),
        task_scope_id="task-scope-1",
        application_id="app-a",
        application_environment_id="env-a",
        principal=_principal(),
    )


def test_governance_request_has_no_physical_agent_fields() -> None:
    fields = MultiAgentCoordinationGovernanceRequest.model_fields
    forbidden = frozenset(
        {
            "agent_id",
            "agent_instance_id",
            "lease_id",
            "execution_id",
            "budget",
            "parent_execution_authority",
        },
    )
    assert forbidden.isdisjoint(fields.keys())


def test_governance_request_uses_multi_agent_evaluation_point() -> None:
    request = _request()
    assert (
        request.evaluation_point is GovernanceEvaluationPoint.MULTI_AGENT_COORDINATION
    )


def test_request_digest_is_sha256() -> None:
    digest = multi_agent_coordination_governance_request_digest(_request())
    assert digest.startswith("sha256:")


def test_evidence_reuses_policy_decision_identity() -> None:
    decision = PolicyDecision(
        action=PolicyAction.ALLOW,
        reason="allow",
        policy_rule_id="rule-1",
        decision_id="decision-1",
    )
    request = _request()
    digest = multi_agent_coordination_governance_request_digest(request)
    evidence = evidence_from_request_and_decision(
        request,
        decision=decision,
        request_digest=digest,
    )
    assert evidence.policy_decision_id == "decision-1"
    assert evidence.policy_action is PolicyAction.ALLOW


def test_policy_rule_rejects_empty_rule_id() -> None:
    with pytest.raises(ValueError, match="rule_id"):
        MultiAgentCoordinationGovernancePolicyRule(
            rule_id=" ",
            decision=PolicyAction.ALLOW,
        )

# © Artur Czarnecki. All rights reserved.

"""NPSC-5D/R1 — runtime multi-agent coordination governance tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.multi_agent_coordination_governance import (
    MultiAgentCoordinationCapabilityKind,
    MultiAgentCoordinationExecutionMode,
    MultiAgentCoordinationGovernanceContribution,
    MultiAgentCoordinationGovernancePolicyRule,
    MultiAgentCoordinationGovernanceRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.governance.multi_agent_coordination_governance import (
    AllowingMultiAgentCoordinationGovernance,
    DenyingMultiAgentCoordinationGovernance,
    FailClosedMultiAgentCoordinationGovernanceEvaluator,
    MultiAgentCoordinationGovernanceBoundary,
    RequireHumanMultiAgentCoordinationGovernance,
    RuntimeMultiAgentCoordinationGovernance,
    UnavailableMultiAgentCoordinationGovernance,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

pytestmark = pytest.mark.unit


def _request() -> MultiAgentCoordinationGovernanceRequest:
    return MultiAgentCoordinationGovernanceRequest(
        intent_id="intent-1",
        execution_mode=MultiAgentCoordinationExecutionMode.SINGLE,
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
        principal=RequestIdentity(
            tenant_id="tenant-a",
            user_id="user-1",
            auth_subject="subject-1",
        ),
    )


def test_allowing_adapter_permits_without_widening_authority() -> None:
    request = _request()
    before = request.model_dump(mode="json")
    result = AllowingMultiAgentCoordinationGovernance().evaluate(request)
    after = request.model_dump(mode="json")
    assert before == after
    assert result.permitted is True
    assert result.decision.action is PolicyAction.ALLOW


def test_denying_adapter_denies() -> None:
    result = DenyingMultiAgentCoordinationGovernance().evaluate(_request())
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY


def test_require_human_adapter_requires_continuation() -> None:
    result = RequireHumanMultiAgentCoordinationGovernance().evaluate(_request())
    assert result.permitted is False
    assert result.requires_governed_continuation is True
    assert result.decision.action is PolicyAction.REQUIRE_HUMAN


def test_unavailable_adapter_fail_closed() -> None:
    result = UnavailableMultiAgentCoordinationGovernance().evaluate(_request())
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY


def test_runtime_evaluator_unconfigured_fail_closed() -> None:
    governance = RuntimeMultiAgentCoordinationGovernance(
        policy_engine=RuntimePolicyEngine(),
    )
    result = governance.evaluate(_request())
    assert result.permitted is False
    assert result.decision.reason == "multi_agent_coordination_unconfigured"


def test_runtime_evaluator_allow_by_mode() -> None:
    governance = RuntimeMultiAgentCoordinationGovernance(
        policy_engine=RuntimePolicyEngine(
            multi_agent_coordination_rules=(
                MultiAgentCoordinationGovernancePolicyRule(
                    rule_id="coordination.single.allow",
                    decision=PolicyAction.ALLOW,
                    execution_mode=MultiAgentCoordinationExecutionMode.SINGLE,
                ),
            ),
        ),
    )
    result = governance.evaluate(_request())
    assert result.permitted is True


def test_modify_maps_to_fail_closed_deny() -> None:
    boundary = MultiAgentCoordinationGovernanceBoundary(
        evaluator=FailClosedMultiAgentCoordinationGovernanceEvaluator(
            policy_engine=RuntimePolicyEngine(),
        ),
    )
    result = boundary.evaluate(_request())
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY

# © Artur Czarnecki. All rights reserved.

"""Runtime execution policy admission tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.runtime_execution_policy_admission import (
    RootExecutionAdmissionPolicyRule,
    RuntimeExecutionPolicyAdmissionRequest,
    WORKER_ROOT_EXECUTION_OPERATION,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    AllowingRuntimeExecutionPolicyAdmission,
    DenyingRuntimeExecutionPolicyAdmission,
    RuntimeExecutionPolicyAdmissionEvaluator,
    UnavailableRuntimeExecutionPolicyAdmission,
    default_worker_root_execution_policy_engine,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

pytestmark = pytest.mark.unit


def _request() -> RuntimeExecutionPolicyAdmissionRequest:
    return RuntimeExecutionPolicyAdmissionRequest(
        tenant_id="tenant-a",
        workspace_id="workspace-x",
        principal_id="principal-1",
        collaborative_authority_scopes=("workspace.read", "workspace.write"),
    )


def test_allowing_adapter_returns_runtime_allow() -> None:
    result = AllowingRuntimeExecutionPolicyAdmission().evaluate(_request())
    assert result.policy_decision.action is PolicyAction.ALLOW


def test_denying_adapter_returns_runtime_deny() -> None:
    result = DenyingRuntimeExecutionPolicyAdmission().evaluate(_request())
    assert result.policy_decision.action is PolicyAction.DENY


def test_unavailable_adapter_fail_closed() -> None:
    result = UnavailableRuntimeExecutionPolicyAdmission().evaluate(_request())
    assert result.policy_decision.action is PolicyAction.DENY
    assert result.policy_decision.reason == "runtime_execution_policy_unavailable"


def test_evaluator_unconfigured_fail_closed() -> None:
    evaluator = RuntimeExecutionPolicyAdmissionEvaluator(
        policy_engine=RuntimePolicyEngine(),
    )
    result = evaluator.evaluate(_request())
    assert result.policy_decision.action is PolicyAction.DENY
    assert result.policy_decision.reason == "root_execution_admission_unconfigured"


def test_evaluator_allow_with_scope_narrowing() -> None:
    evaluator = RuntimeExecutionPolicyAdmissionEvaluator(
        policy_engine=RuntimePolicyEngine(
            root_execution_admission_rules=(
                RootExecutionAdmissionPolicyRule(
                    rule_id="runtime.read_only",
                    decision=PolicyAction.ALLOW,
                    execution_operation=WORKER_ROOT_EXECUTION_OPERATION,
                    approved_scopes=("workspace.read",),
                ),
            ),
        ),
    )
    result = evaluator.evaluate(_request())
    assert result.policy_decision.action is PolicyAction.ALLOW
    assert result.approved_scopes == ("workspace.read",)


def test_default_worker_root_execution_policy_engine_allows() -> None:
    evaluator = RuntimeExecutionPolicyAdmissionEvaluator(
        policy_engine=default_worker_root_execution_policy_engine(),
    )
    result = evaluator.evaluate(_request())
    assert result.policy_decision.action is PolicyAction.ALLOW

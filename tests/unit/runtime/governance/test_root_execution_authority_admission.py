# © Artur Czarnecki. All rights reserved.

"""Runtime/Governance root execution authority admission tests."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from intergrax.contracts.collaborative_work import EffectiveAuthorityDecision
from intergrax.contracts.runtime_execution_admission import (
    RootExecutionAuthorityAdmissionDisposition,
    RootExecutionAuthorityAdmissionRequest,
)
from intergrax.contracts.runtime_execution_policy_admission import (
    RootExecutionAdmissionPolicyRule,
    WORKER_ROOT_EXECUTION_OPERATION,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.root_execution_authority_admission import (
    DenyingRootExecutionAuthorityAdmission,
    RootExecutionAuthorityAdmissionService,
    UnavailableRootExecutionAuthorityAdmission,
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    AllowingRuntimeExecutionPolicyAdmission,
    DenyingRuntimeExecutionPolicyAdmission,
    RequireHumanRuntimeExecutionPolicyAdmission,
    RuntimeExecutionPolicyAdmissionEvaluator,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

pytestmark = pytest.mark.unit


def _request(*, action: PolicyAction) -> RootExecutionAuthorityAdmissionRequest:
    return RootExecutionAuthorityAdmissionRequest(
        tenant_id="tenant-a",
        workspace_id="workspace-x",
        principal_id="principal-1",
        collaborative_authority_scopes=("workspace.read",),
        effective_authority_decision=EffectiveAuthorityDecision(
            decision=PolicyDecision(action=action, reason="test"),
        ),
    )


def test_default_service_fails_closed_on_collaborative_allow() -> None:
    service = RootExecutionAuthorityAdmissionService()
    result = service.authorize(_request(action=PolicyAction.ALLOW))
    assert result.disposition in {
        RootExecutionAuthorityAdmissionDisposition.UNAVAILABLE,
        RootExecutionAuthorityAdmissionDisposition.DENIED,
    }
    assert result.trusted_parent_execution_authority is None
    assert result.policy_decision.reason == "root_execution_admission_unconfigured"


def test_explicit_allow_policy_mints_trusted_authority() -> None:
    service = RootExecutionAuthorityAdmissionService(
        runtime_policy_admission=RuntimeExecutionPolicyAdmissionEvaluator(
            policy_engine=RuntimePolicyEngine(
                root_execution_admission_rules=(
                    RootExecutionAdmissionPolicyRule(
                        rule_id="configured.worker.root_execution.allow",
                        decision=PolicyAction.ALLOW,
                        execution_operation=WORKER_ROOT_EXECUTION_OPERATION,
                    ),
                ),
            ),
        ),
    )
    result = service.authorize(_request(action=PolicyAction.ALLOW))
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.ALLOWED
    assert result.trusted_parent_execution_authority is not None
    assert result.policy_decision.policy_rule_id == "configured.worker.root_execution.allow"


def test_explicit_deny_policy_does_not_mint_authority() -> None:
    service = RootExecutionAuthorityAdmissionService(
        runtime_policy_admission=RuntimeExecutionPolicyAdmissionEvaluator(
            policy_engine=RuntimePolicyEngine(
                root_execution_admission_rules=(
                    RootExecutionAdmissionPolicyRule(
                        rule_id="configured.worker.root_execution.deny",
                        decision=PolicyAction.DENY,
                        execution_operation=WORKER_ROOT_EXECUTION_OPERATION,
                    ),
                ),
            ),
        ),
    )
    result = service.authorize(_request(action=PolicyAction.ALLOW))
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.DENIED
    assert result.trusted_parent_execution_authority is None
    assert result.policy_decision.policy_rule_id == "configured.worker.root_execution.deny"


def test_no_matching_rule_fails_closed() -> None:
    service = RootExecutionAuthorityAdmissionService(
        runtime_policy_admission=RuntimeExecutionPolicyAdmissionEvaluator(
            policy_engine=RuntimePolicyEngine(
                root_execution_admission_rules=(
                    RootExecutionAdmissionPolicyRule(
                        rule_id="configured.other.operation.allow",
                        decision=PolicyAction.ALLOW,
                        execution_operation="other.operation",
                    ),
                ),
            ),
        ),
    )
    result = service.authorize(_request(action=PolicyAction.ALLOW))
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.DENIED
    assert result.trusted_parent_execution_authority is None
    assert result.policy_decision.reason == "root_execution_admission_indeterminate"


def test_empty_rule_set_fails_closed() -> None:
    service = RootExecutionAuthorityAdmissionService(
        runtime_policy_admission=RuntimeExecutionPolicyAdmissionEvaluator(
            policy_engine=RuntimePolicyEngine(),
        ),
    )
    result = service.authorize(_request(action=PolicyAction.ALLOW))
    assert result.disposition in {
        RootExecutionAuthorityAdmissionDisposition.UNAVAILABLE,
        RootExecutionAuthorityAdmissionDisposition.DENIED,
    }
    assert result.trusted_parent_execution_authority is None
    assert result.policy_decision.reason == "root_execution_admission_unconfigured"


def test_allow_mints_scoped_trusted_authority() -> None:
    service = RootExecutionAuthorityAdmissionService(
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    result = service.authorize(_request(action=PolicyAction.ALLOW))
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.ALLOWED
    assert result.trusted_parent_execution_authority is not None
    assert result.trusted_parent_execution_authority.permission_scopes == (
        "workspace.read",
    )


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (PolicyAction.DENY, RootExecutionAuthorityAdmissionDisposition.DENIED),
        (PolicyAction.REQUIRE_HUMAN, RootExecutionAuthorityAdmissionDisposition.REQUIRE_HUMAN),
        (PolicyAction.ESCALATE, RootExecutionAuthorityAdmissionDisposition.ESCALATE),
        (PolicyAction.MODIFY, RootExecutionAuthorityAdmissionDisposition.DENIED),
    ],
)
def test_non_allow_fail_closed(
    action: PolicyAction,
    expected: RootExecutionAuthorityAdmissionDisposition,
) -> None:
    service = RootExecutionAuthorityAdmissionService(
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    result = service.authorize(_request(action=action))
    assert result.disposition is expected
    assert result.trusted_parent_execution_authority is None


def test_collaborative_allow_runtime_deny_does_not_mint_authority() -> None:
    service = RootExecutionAuthorityAdmissionService(
        runtime_policy_admission=DenyingRuntimeExecutionPolicyAdmission(),
    )
    result = service.authorize(_request(action=PolicyAction.ALLOW))
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.DENIED
    assert result.trusted_parent_execution_authority is None


def test_collaborative_allow_runtime_require_human_does_not_mint_authority() -> None:
    service = RootExecutionAuthorityAdmissionService(
        runtime_policy_admission=RequireHumanRuntimeExecutionPolicyAdmission(),
    )
    result = service.authorize(_request(action=PolicyAction.ALLOW))
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.REQUIRE_HUMAN
    assert result.trusted_parent_execution_authority is None


def test_collaborative_allow_runtime_allow_mints_authority() -> None:
    service = RootExecutionAuthorityAdmissionService(
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    result = service.authorize(_request(action=PolicyAction.ALLOW))
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.ALLOWED
    assert result.trusted_parent_execution_authority is not None


def test_denying_adapter() -> None:
    service = DenyingRootExecutionAuthorityAdmission()
    result = service.authorize(_request(action=PolicyAction.ALLOW))
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.DENIED


def test_unavailable_adapter() -> None:
    service = UnavailableRootExecutionAuthorityAdmission()
    result = service.authorize(_request(action=PolicyAction.ALLOW))
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.UNAVAILABLE


def test_root_admission_invokes_independent_runtime_policy_port() -> None:
    module = importlib.import_module(
        "intergrax.runtime.governance.root_execution_authority_admission",
    )
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    authorize_source = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "authorize":
            authorize_source = ast.get_source_segment(source, node) or ""
            break
    assert "_runtime_policy_admission.evaluate" in authorize_source
    assert "collaborative_decision.action is not PolicyAction.ALLOW" in authorize_source
    mint_lines = [
        line
        for line in authorize_source.splitlines()
        if "ParentExecutionAuthority.scoped" in line
    ]
    assert len(mint_lines) == 1
    mint_index = authorize_source.index("ParentExecutionAuthority.scoped")
    evaluate_index = authorize_source.index("_runtime_policy_admission.evaluate")
    assert evaluate_index < mint_index

# © Artur Czarnecki. All rights reserved.

"""Canonical bundle-backed control-plane mutation policy adapter tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.applications._shared.task_control_governance import (
    MUTATION_TYPE_CANCEL_TASK_EXECUTION,
    build_cancel_task_execution_mutation_request,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.contracts.runtime_policy_bundle import (
    PolicyBundleRule,
    build_immutable_runtime_policy_bundle,
)
from intergrax.runtime.capacity.control_plane_governance import (
    MUTATION_TYPE_SCALE_K8S_DEPLOYMENT,
    build_scale_k8s_deployment_mutation_request,
)
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.governance.control_plane_mutation_policy import (
    BundleBackedControlPlaneMutationEvaluator,
    control_plane_mutation_bundle_match_input,
)
from intergrax.runtime.policy.runtime_policy_bundle_evaluator import (
    RuntimePolicyBundleEvaluator,
)
from intergrax.runtime.task.task import TaskState

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_T0 = datetime(2026, 8, 24, 0, 0, 0, tzinfo=timezone.utc)
_TENANT = "tenant-policy-adapter"


def _principal(*, user_id: str = "operator-1") -> RequestIdentity:
    return RequestIdentity(
        tenant_id=_TENANT,
        user_id=user_id,
        principal_type=PrincipalType.USER,
        auth_subject=user_id,
    )


def _bundle(*rules: PolicyBundleRule):
    return build_immutable_runtime_policy_bundle(
        bundle_id="cpm-adapter-pack",
        version="1.0.0",
        rules=rules,
        issued_at=_T0,
    )


def test_control_plane_mutation_maps_mutation_type_to_match_action() -> None:
    request = build_cancel_task_execution_mutation_request(
        principal=_principal(),
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        mutation_id="mut-1",
        current_state=TaskState.RUNNING,
    )
    match_input = control_plane_mutation_bundle_match_input(request)
    assert match_input.action == MUTATION_TYPE_CANCEL_TASK_EXECUTION
    assert match_input.principal_id == "operator-1"
    assert match_input.tenant_id == _TENANT
    assert match_input.context["task_id"] is not None
    assert match_input.context["run_id"] is not None


def test_standalone_ecp_scale_maps_without_execution_identity() -> None:
    request = build_scale_k8s_deployment_mutation_request(
        principal=_principal(user_id="scheduler"),
        tenant_id=_TENANT,
        mutation_id="mut-scale",
        deployment="nexus-host",
        current_replicas=2,
        target_replicas=3,
    )
    match_input = control_plane_mutation_bundle_match_input(request)
    assert match_input.action == MUTATION_TYPE_SCALE_K8S_DEPLOYMENT
    assert match_input.context["task_id"] is None
    assert match_input.context["run_id"] is None


def test_bundle_backed_evaluator_allow_on_explicit_match() -> None:
    bundle = _bundle(
        PolicyBundleRule(
            rule_id="task_control.cancel",
            match_action=MUTATION_TYPE_CANCEL_TASK_EXECUTION,
            effect="allow",
        ),
    )
    bundle_evaluator = RuntimePolicyBundleEvaluator(bundle, clock=lambda: _T0)
    evaluator = BundleBackedControlPlaneMutationEvaluator(bundle_evaluator=bundle_evaluator)
    request = build_cancel_task_execution_mutation_request(
        principal=_principal(user_id="caller-42"),
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        mutation_id="mut-allow",
        current_state=TaskState.RUNNING,
    )
    decision = evaluator.evaluate(request)
    assert decision.action is PolicyAction.ALLOW
    assert decision.policy_rule_id == "task_control.cancel"
    assert bundle_evaluator.match_calls[-1].principal_id == "caller-42"


def test_bundle_backed_evaluator_standalone_ecp_allow_without_execution_identity() -> None:
    bundle = _bundle(
        PolicyBundleRule(
            rule_id="ecp.scale_k8s",
            match_action=MUTATION_TYPE_SCALE_K8S_DEPLOYMENT,
            effect="allow",
        ),
    )
    bundle_evaluator = RuntimePolicyBundleEvaluator(bundle, clock=lambda: _T0)
    evaluator = BundleBackedControlPlaneMutationEvaluator(bundle_evaluator=bundle_evaluator)
    request = build_scale_k8s_deployment_mutation_request(
        principal=_principal(user_id="capacity-scheduler"),
        tenant_id=_TENANT,
        mutation_id="mut-ecp-allow",
        deployment="nexus-host",
        current_replicas=2,
        target_replicas=3,
    )
    decision = evaluator.evaluate(request)
    assert decision.action is PolicyAction.ALLOW
    assert decision.policy_rule_id == "ecp.scale_k8s"


def test_bundle_backed_evaluator_fail_closed_without_match() -> None:
    bundle = _bundle(
        PolicyBundleRule(
            rule_id="other.only",
            match_action="other.mutation",
            effect="allow",
        ),
    )
    evaluator = BundleBackedControlPlaneMutationEvaluator(
        bundle_evaluator=RuntimePolicyBundleEvaluator(bundle, clock=lambda: _T0),
    )
    request = build_cancel_task_execution_mutation_request(
        principal=_principal(),
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        mutation_id="mut-deny",
        current_state=TaskState.RUNNING,
    )
    decision = evaluator.evaluate(request)
    assert decision.action is PolicyAction.DENY
    assert decision.policy_rule_id == "bundle.no_match"


def test_bundle_backed_evaluator_standalone_ecp_explicit_deny() -> None:
    bundle = _bundle(
        PolicyBundleRule(
            rule_id="ecp.scale_k8s",
            match_action=MUTATION_TYPE_SCALE_K8S_DEPLOYMENT,
            effect="deny",
        ),
    )
    evaluator = BundleBackedControlPlaneMutationEvaluator(
        bundle_evaluator=RuntimePolicyBundleEvaluator(bundle, clock=lambda: _T0),
    )
    request = build_scale_k8s_deployment_mutation_request(
        principal=_principal(),
        tenant_id=_TENANT,
        mutation_id="mut-ecp-deny",
        deployment="nexus-host",
        current_replicas=2,
        target_replicas=3,
    )
    decision = evaluator.evaluate(request)
    assert decision.action is PolicyAction.DENY


def test_bundle_backed_evaluator_require_human_from_explicit_rule() -> None:
    bundle = _bundle(
        PolicyBundleRule(
            rule_id="task_control.cancel_human",
            match_action=MUTATION_TYPE_CANCEL_TASK_EXECUTION,
            effect="require_human",
        ),
    )
    evaluator = BundleBackedControlPlaneMutationEvaluator(
        bundle_evaluator=RuntimePolicyBundleEvaluator(bundle, clock=lambda: _T0),
    )
    request = build_cancel_task_execution_mutation_request(
        principal=_principal(),
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        mutation_id="mut-human",
        current_state=TaskState.RUNNING,
    )
    decision = evaluator.evaluate(request)
    assert decision.action is PolicyAction.REQUIRE_HUMAN
    assert decision.policy_rule_id == "task_control.cancel_human"


def test_boundary_fail_closed_on_evaluator_exception() -> None:
    class _BrokenEvaluator:
        def evaluate(self, request: object) -> PolicyDecision:
            raise RuntimeError("policy backend unavailable")

    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=_BrokenEvaluator())
    request = build_scale_k8s_deployment_mutation_request(
        principal=_principal(),
        tenant_id=_TENANT,
        mutation_id="mut-exc",
        deployment="nexus-host",
        current_replicas=2,
        target_replicas=3,
    )
    result = boundary.authorize(request)
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY

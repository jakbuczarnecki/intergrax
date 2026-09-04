# © Artur Czarnecki. All rights reserved.

"""TASK-CPM-1C — canonical explicit policy authority proofs (TASKCPM-P1–P12)."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.agent_distribution.control_plane_governance import build_activation_mutation_request
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.applications._shared.harness_control_plane_governance_wiring import (
    build_harness_control_plane_governance,
    resolve_harness_task_control_mutation_boundary,
)
from intergrax.applications._shared.harness_control_plane_policy_wiring import (
    build_harness_host_control_plane_policy_bundle,
    build_reference_production_lifecycle_policy_bundle,
)
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.harness_task_routes import mount_harness_task_routes
from intergrax.applications._shared.reference_production_governance_wiring import (
    build_reference_production_control_plane_governance,
)
from intergrax.applications._shared.task_control_governance import (
    MUTATION_TYPE_CANCEL_TASK_EXECUTION,
    build_cancel_task_execution_mutation_request,
)
from intergrax.applications._shared.task_control_wiring import wire_harness_task_control
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications._shared.harness_host_runtime_compat import resolve_harness_host_nexus_loop_legacy
from governed_contractor_application.tests.governed_contractor_ac3_projection import (
    build_governed_contractor_test_registry_projection,
)
from governed_contractor_application.manifest import build_governed_contractor_manifest
from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRequest
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision
from intergrax.contracts.runtime_policy_bundle import (
    PolicyBundleRule,
    build_immutable_runtime_policy_bundle,
)
from intergrax.integrations.contracts.identity_provider import IdentityUser
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.governance.control_plane_mutation_approval import (
    ApprovalConsumingControlPlaneMutationEvaluator,
    ControlPlaneMutationApprovalCoordinator,
)
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.governance.control_plane_mutation_policy import (
    BundleBackedControlPlaneMutationEvaluator,
)
from intergrax.runtime.policy.runtime_policy_bundle_evaluator import RuntimePolicyBundleEvaluator
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-policy-authority"
_MUTATION_ID = "mut-policy-cancel-1"
_T0 = build_harness_host_control_plane_policy_bundle().issued_at


def _variant_bundle(*rules: PolicyBundleRule):
    return build_immutable_runtime_policy_bundle(
        bundle_id="harness.control_plane",
        version="1.0.0-test",
        rules=rules,
        issued_at=_T0,
    )


@pytest.fixture
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(
        env: object,
        agent_override: object | None = None,
        **_: object,
    ) -> object:
        del env
        if agent_override is not None:
            return agent_override
        return adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )


class _FakeIdentityProvider:
    def verify_token(self, token: str) -> IdentityUser:
        if token != "valid-bearer":
            raise ValueError("invalid")
        return IdentityUser(user_id="operator-1", tenant_id=_TENANT)


@dataclass
class _RecordingEvaluator:
    decision: PolicyDecision = field(
        default_factory=lambda: PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="test_allow",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="task_control.test_allow",
            decision_id="dec-allow",
        )
    )
    calls: list[ControlPlaneMutationRequest] = field(default_factory=list)

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        self.calls.append(request)
        return self.decision


def _principal(*, tenant_id: str = _TENANT, user_id: str = "operator-1") -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id=user_id,
        principal_type=PrincipalType.USER,
        auth_subject=user_id,
    )


def _task(*, tenant_id: str = _TENANT) -> Task:
    return Task(
        task_id=mint_task_id(),
        tenant_id=tenant_id,
        user_id="user-1",
        message="hello",
        context=TaskContext(),
        state=TaskState.RUNNING,
    )


def _product_runtime(
    *,
    mutation_boundary: ControlPlaneMutationAuthorizationBoundary | None = None,
) -> object:
    settings = GovernedContractorBackendSettings.from_env()
    manifest = build_governed_contractor_manifest()
    env = manifest.environment or build_governed_contractor_environment_profile(settings)
    return build_harness_host_runtime(
        manifest,
        env,
        mutation_authorization_boundary=mutation_boundary,
        registry_projection=build_governed_contractor_test_registry_projection(),
    )


def _cancel_request(*, principal: RequestIdentity | None = None) -> ControlPlaneMutationRequest:
    return build_cancel_task_execution_mutation_request(
        principal=principal or _principal(),
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        mutation_id=_MUTATION_ID,
        current_state=TaskState.RUNNING,
    )


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    ActiveTaskRegistry.clear_for_tests()


def test_taskcpm_p1_product_cancel_allow_only_with_explicit_rule() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_TENANT)
    boundary = resolve_harness_task_control_mutation_boundary(
        build_harness_control_plane_governance(env),
    )
    assert boundary is not None
    result = boundary.authorize(_cancel_request())
    assert result.permitted is True
    assert result.decision.policy_rule_id == "harness.task_control.cancel_task_execution"


def test_taskcpm_p2_unmatched_cancel_rule_denies() -> None:
    empty_bundle = _variant_bundle()
    boundary = ControlPlaneMutationAuthorizationBoundary(
        evaluator=BundleBackedControlPlaneMutationEvaluator(
            bundle_evaluator=RuntimePolicyBundleEvaluator(empty_bundle),
        ),
    )
    result = boundary.authorize(_cancel_request())
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY


@pytest.mark.asyncio
async def test_taskcpm_p2b_unmatched_cancel_zero_side_effect(_stub_host_llm: None) -> None:
    empty_bundle = _variant_bundle()
    deny_boundary = ControlPlaneMutationAuthorizationBoundary(
        evaluator=BundleBackedControlPlaneMutationEvaluator(
            bundle_evaluator=RuntimePolicyBundleEvaluator(empty_bundle),
        ),
    )
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    runtime = _product_runtime(mutation_boundary=deny_boundary)
    app = FastAPI()
    app.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(),
        require_api_key=False,
        resolved_api_key=None,
        tenant_required=True,
    )
    wire_harness_task_control(
        app,
        enabled=True,
        task_runner=UnifiedTaskRunner(resolve_harness_host_nexus_loop_legacy(runtime)),  # type: ignore[arg-type]
        env=runtime.environment,
        runtime=runtime,
    )
    client = TestClient(app)
    with patch(
        "intergrax.applications._shared.task_control.CancellationCoordinator.request",
    ) as cancel_request:
        response = client.post(
            f"/v1/tasks/{task.task_id}/cancel",
            json={"mutation_id": _MUTATION_ID, "run_id": str(run_id)},
            headers={"Authorization": "Bearer valid-bearer"},
        )
    assert response.status_code == 403
    assert cancel_request.call_count == 0


@pytest.mark.asyncio
async def test_taskcpm_p3_explicit_deny_zero_cancel_effect(_stub_host_llm: None) -> None:
    deny_bundle = _variant_bundle(
        PolicyBundleRule(
            rule_id="harness.task_control.cancel_task_execution",
            match_action=MUTATION_TYPE_CANCEL_TASK_EXECUTION,
            effect="deny",
        ),
    )
    deny_boundary = ControlPlaneMutationAuthorizationBoundary(
        evaluator=BundleBackedControlPlaneMutationEvaluator(
            bundle_evaluator=RuntimePolicyBundleEvaluator(deny_bundle),
        ),
    )
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    runtime = _product_runtime(mutation_boundary=deny_boundary)
    app = FastAPI()
    app.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(),
        require_api_key=False,
        resolved_api_key=None,
        tenant_required=True,
    )
    wire_harness_task_control(
        app,
        enabled=True,
        task_runner=UnifiedTaskRunner(resolve_harness_host_nexus_loop_legacy(runtime)),  # type: ignore[arg-type]
        env=runtime.environment,
        runtime=runtime,
    )
    client = TestClient(app)
    with patch(
        "intergrax.applications._shared.task_control.CancellationCoordinator.request",
    ) as cancel_request:
        response = client.post(
            f"/v1/tasks/{task.task_id}/cancel",
            json={"mutation_id": _MUTATION_ID, "run_id": str(run_id)},
            headers={"Authorization": "Bearer valid-bearer"},
        )
    assert response.status_code == 403
    assert cancel_request.call_count == 0


@pytest.mark.asyncio
async def test_taskcpm_p4_require_human_zero_cancel_with_evidence(_stub_host_llm: None) -> None:
    human_bundle = _variant_bundle(
        PolicyBundleRule(
            rule_id="harness.task_control.cancel_task_execution",
            match_action=MUTATION_TYPE_CANCEL_TASK_EXECUTION,
            effect="require_human",
        ),
    )
    human_boundary = ControlPlaneMutationAuthorizationBoundary(
        evaluator=BundleBackedControlPlaneMutationEvaluator(
            bundle_evaluator=RuntimePolicyBundleEvaluator(human_bundle),
        ),
    )
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    runtime = _product_runtime(mutation_boundary=human_boundary)
    app = FastAPI()
    app.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(),
        require_api_key=False,
        resolved_api_key=None,
        tenant_required=True,
    )
    wire_harness_task_control(
        app,
        enabled=True,
        task_runner=UnifiedTaskRunner(resolve_harness_host_nexus_loop_legacy(runtime)),  # type: ignore[arg-type]
        env=runtime.environment,
        runtime=runtime,
    )
    client = TestClient(app)
    with patch(
        "intergrax.applications._shared.task_control.CancellationCoordinator.request",
    ) as cancel_request:
        response = client.post(
            f"/v1/tasks/{task.task_id}/cancel",
            json={"mutation_id": _MUTATION_ID, "run_id": str(run_id)},
            headers={"Authorization": "Bearer valid-bearer"},
        )
    assert response.status_code == 403
    detail = response.json()["detail"]
    assert detail["blocker_code"] == "TASK_CONTROL_BLOCKED_BY_REQUIRE_HUMAN"
    assert detail["policy_action"] == PolicyAction.REQUIRE_HUMAN.value
    assert "authorization_evidence" in detail
    assert cancel_request.call_count == 0


def test_taskcpm_p5_evaluator_receives_http_caller_identity() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_TENANT)
    boundary = resolve_harness_task_control_mutation_boundary(
        build_harness_control_plane_governance(env),
    )
    assert boundary is not None
    caller = _principal(user_id="http-operator-77")
    service = build_reference_production_control_plane_governance(env).principal
    assert service.user_id != caller.user_id
    boundary.authorize(_cancel_request(principal=caller))
    evaluator = boundary.evaluator
    assert isinstance(evaluator, ApprovalConsumingControlPlaneMutationEvaluator)
    inner = evaluator.inner
    assert isinstance(inner, BundleBackedControlPlaneMutationEvaluator)
    assert inner.bundle_evaluator.calls[-1].principal_id == "http-operator-77"


def test_taskcpm_p6_policy_evaluates_exact_cancel_mutation_type() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_TENANT)
    boundary = resolve_harness_task_control_mutation_boundary(
        build_harness_control_plane_governance(env),
    )
    assert boundary is not None
    request = _cancel_request()
    assert request.mutation_type == MUTATION_TYPE_CANCEL_TASK_EXECUTION
    result = boundary.authorize(request)
    assert result.evidence.mutation_type == MUTATION_TYPE_CANCEL_TASK_EXECUTION
    assert result.decision.policy_rule_id == "harness.task_control.cancel_task_execution"


def test_taskcpm_p7_lifecycle_policy_does_not_authorize_cancel() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_TENANT)
    reference = build_reference_production_control_plane_governance(env)
    cancel_result = reference.mutation_authorization_boundary.authorize(_cancel_request())
    assert cancel_result.permitted is False
    assert cancel_result.decision.action is PolicyAction.DENY


def test_taskcpm_p8_cancel_policy_does_not_authorize_unrelated_mutation() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_TENANT)
    boundary = resolve_harness_task_control_mutation_boundary(
        build_harness_control_plane_governance(env),
    )
    assert boundary is not None
    lifecycle_request = build_activation_mutation_request(
        principal=_principal(),
        application_id="app-1",
        application_environment_id="env-1",
        mutation_id="mut-activate",
        current_traffic_revision_id=None,
        current_serving_pointer_revision=0,
        target_runtime_revision_id="rev-1",
    )
    result = boundary.authorize(lifecycle_request)
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY


def test_taskcpm_p9_approval_consuming_evaluator_preserves_scoped_approval() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_TENANT)
    governance = build_harness_control_plane_governance(env)
    boundary = governance.mutation_authorization_boundary
    assert boundary is not None
    assert isinstance(boundary.evaluator, ApprovalConsumingControlPlaneMutationEvaluator)
    assert governance.approval_coordinator is not None


def test_taskcpm_p10_product_host_uses_canonical_bundle_policy_authority(
    _stub_host_llm: None,
) -> None:
    runtime = _product_runtime()
    boundary = resolve_harness_task_control_mutation_boundary(runtime.control_plane_governance)
    assert boundary is not None
    evaluator = boundary.evaluator
    assert isinstance(evaluator, ApprovalConsumingControlPlaneMutationEvaluator)
    inner = evaluator.inner
    assert isinstance(inner, BundleBackedControlPlaneMutationEvaluator)
    assert inner.bundle_evaluator.bundle.bundle_id == "harness.control_plane"


def test_taskcpm_p11_lab_missing_boundary_fail_closed() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id=_TENANT)
    assert env.application_profile is ApplicationProfile.LAB
    governance = build_harness_control_plane_governance(env)
    assert resolve_harness_task_control_mutation_boundary(governance) is None


@pytest.mark.asyncio
async def test_taskcpm_p11b_lab_direct_mount_fail_closed() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    app = FastAPI()
    app.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(),
        require_api_key=False,
        resolved_api_key=None,
        tenant_required=True,
    )
    mount_harness_task_routes(
        app,
        task_runner=UnifiedTaskRunner(object()),  # type: ignore[arg-type]
        mutation_boundary=None,
    )
    client = TestClient(app)
    with patch(
        "intergrax.applications._shared.task_control.CancellationCoordinator.request",
    ) as cancel_request:
        response = client.post(
            f"/v1/tasks/{task.task_id}/cancel",
            json={"mutation_id": _MUTATION_ID, "run_id": str(run_id)},
            headers={"Authorization": "Bearer valid-bearer"},
        )
    assert response.status_code == 403
    assert response.json()["detail"]["blocker_code"] == "TASK_CONTROL_BLOCKED_BY_MISSING_BOUNDARY"
    assert cancel_request.call_count == 0


def test_taskcpm_p12_no_explicit_rule_conservative_failure() -> None:
    bundle = build_reference_production_lifecycle_policy_bundle()
    assert all(
        rule.match_action != MUTATION_TYPE_CANCEL_TASK_EXECUTION
        for rule in bundle.rules
    )
    boundary = ControlPlaneMutationAuthorizationBoundary(
        evaluator=BundleBackedControlPlaneMutationEvaluator(
            bundle_evaluator=RuntimePolicyBundleEvaluator(bundle),
        ),
    )
    result = boundary.authorize(_cancel_request())
    assert result.permitted is False
    assert result.decision.policy_rule_id == "bundle.no_match"

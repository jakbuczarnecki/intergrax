# © Artur Czarnecki. All rights reserved.

"""TASK-CPM-1B — canonical host boundary composition proofs (TASKCPM-H1–H8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.agent_distribution.control_plane_governance import build_activation_mutation_request
from governed_contractor_application.tests.governed_contractor_ac3_projection import (
    build_governed_contractor_test_registry_projection,
)
from governed_contractor_application.host.factory import create_governed_contractor_backend_app
from governed_contractor_application.manifest import build_governed_contractor_manifest
from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.applications._shared.harness_control_plane_governance_wiring import (
    build_harness_control_plane_governance,
    resolve_harness_task_control_mutation_boundary,
)
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.harness_task_routes import mount_harness_task_routes
from intergrax.runtime.governance.control_plane_mutation_approval import (
    ApprovalConsumingControlPlaneMutationEvaluator,
)
from intergrax.runtime.governance.control_plane_mutation_policy import (
    BundleBackedControlPlaneMutationEvaluator,
)
from intergrax.applications._shared.task_control_governance import (
    build_cancel_task_execution_mutation_request,
)
from intergrax.applications._shared.task_control_wiring import wire_harness_task_control
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRequest
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision
from intergrax.integrations.contracts.identity_provider import IdentityUser
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from intergrax.scaffold.application_names import ScaffoldApplicationNames
from intergrax.scaffold.new_application_product import factory_py
from local_workspace_application.host.factory import create_local_workspace_backend_app
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.tests.lkw_ac3_projection import build_lkw_test_registry_projection

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-host-composition"
_MUTATION_ID = "mut-host-cancel-1"


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


def _principal(*, tenant_id: str = _TENANT) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id="operator-1",
        principal_type=PrincipalType.USER,
        auth_subject="operator-1",
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


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    ActiveTaskRegistry.clear_for_tests()


def test_taskcpm_h1_product_host_runtime_exposes_canonical_boundary(
    _stub_host_llm: None,
) -> None:
    runtime = _product_runtime()
    assert runtime.control_plane_governance is not None
    boundary = resolve_harness_task_control_mutation_boundary(runtime.control_plane_governance)
    assert boundary is not None


@pytest.mark.asyncio
async def test_taskcpm_h2_allow_through_host_composition_reaches_cooperative_cancel(
    _stub_host_llm: None,
) -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    runtime = _product_runtime()
    boundary = resolve_harness_task_control_mutation_boundary(runtime.control_plane_governance)
    assert boundary is not None
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
        task_runner=UnifiedTaskRunner(runtime.nexus_loop),  # type: ignore[arg-type]
        env=runtime.environment,
        runtime=runtime,
    )
    client = TestClient(app)
    with patch(
        "intergrax.applications._shared.task_control.CancellationCoordinator.request",
        wraps=CancellationCoordinator.request,
    ) as cancel_request:
        response = client.post(
            f"/v1/tasks/{task.task_id}/cancel",
            json={
                "mutation_id": _MUTATION_ID,
                "run_id": str(run_id),
                "reason": "operator_requested",
            },
            headers={"Authorization": "Bearer valid-bearer"},
        )
    assert response.status_code == 200
    assert cancel_request.call_count == 1


@pytest.mark.asyncio
async def test_taskcpm_h3_deny_through_host_composed_boundary_zero_cancel_effect(
    _stub_host_llm: None,
) -> None:
    deny_evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.DENY,
            reason="deny_cancel",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="task_control.deny",
            decision_id="dec-deny",
        )
    )
    deny_boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=deny_evaluator)
    runtime = _product_runtime(mutation_boundary=deny_boundary)
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
    wire_harness_task_control(
        app,
        enabled=True,
        task_runner=UnifiedTaskRunner(runtime.nexus_loop),  # type: ignore[arg-type]
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
    assert not CancellationCoordinator.is_requested(task.metadata)


def test_taskcpm_h4_product_host_uses_canonical_bundle_policy_authority() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_TENANT)
    governance = build_harness_control_plane_governance(env)
    boundary = resolve_harness_task_control_mutation_boundary(governance)
    assert boundary is not None
    cancel_request = build_cancel_task_execution_mutation_request(
        principal=_principal(),
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        mutation_id=_MUTATION_ID,
        current_state=TaskState.RUNNING,
    )
    host_result = boundary.authorize(cancel_request)
    assert host_result.permitted is True
    assert host_result.decision.policy_rule_id == "harness.task_control.cancel_task_execution"


@pytest.mark.asyncio
async def test_taskcpm_h5_missing_boundary_remains_fail_closed_on_direct_mount() -> None:
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
    detail = response.json()["detail"]
    assert detail["blocker_code"] == "TASK_CONTROL_BLOCKED_BY_MISSING_BOUNDARY"
    assert cancel_request.call_count == 0


def test_taskcpm_h6_host_policy_separates_cancel_from_lifecycle_mutations() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_TENANT)
    governance = build_harness_control_plane_governance(env)
    boundary = resolve_harness_task_control_mutation_boundary(governance)
    assert boundary is not None
    principal = _principal()
    cancel_request = build_cancel_task_execution_mutation_request(
        principal=principal,
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        mutation_id="mut-cancel-shared",
        current_state=TaskState.RUNNING,
    )
    lifecycle_request = build_activation_mutation_request(
        principal=principal,
        application_id="app-1",
        application_environment_id="env-1",
        mutation_id="mut-activate-shared",
        current_traffic_revision_id=None,
        current_serving_pointer_revision=0,
        target_runtime_revision_id="rev-1",
    )
    cancel_result = boundary.authorize(cancel_request)
    lifecycle_result = boundary.authorize(lifecycle_request)
    assert cancel_result.permitted is True
    assert lifecycle_result.permitted is False
    assert cancel_result.decision.policy_rule_id != lifecycle_result.decision.policy_rule_id


def test_taskcpm_h7_second_product_host_reuses_shared_composition() -> None:
    env_a = ApplicationEnvironmentProfile.product_defaults(profile_id="tenant-a")
    env_b = ApplicationEnvironmentProfile.product_defaults(profile_id="tenant-b")
    boundary_a = resolve_harness_task_control_mutation_boundary(
        build_harness_control_plane_governance(env_a),
    )
    boundary_b = resolve_harness_task_control_mutation_boundary(
        build_harness_control_plane_governance(env_b),
    )
    assert boundary_a is not None
    assert boundary_b is not None
    request = build_cancel_task_execution_mutation_request(
        principal=_principal(tenant_id="tenant-a"),
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        mutation_id=_MUTATION_ID,
        current_state=TaskState.RUNNING,
    )
    assert boundary_a.authorize(request).permitted is True
    assert boundary_b.authorize(request).permitted is True


def test_taskcpm_h8_scaffold_product_factory_uses_canonical_task_control_wiring() -> None:
    names = ScaffoldApplicationNames.resolve("example_application")
    source = factory_py(names)
    assert "wire_harness_task_control" in source
    assert "runtime=runtime" in source
    assert "mount_harness_task_routes" not in source


def test_taskcpm_h1b_governed_contractor_factory_wires_runtime_boundary(
    _stub_host_llm: None,
) -> None:
    settings = GovernedContractorBackendSettings(
        include_task_control=True,
        include_mcp=False,
        include_scheduler=False,
        include_interaction_routes=False,
    )
    app = create_governed_contractor_backend_app(
        registry_projection=build_governed_contractor_test_registry_projection(),
        settings=settings,
    )
    paths = {route.path for route in app.routes}
    assert "/v1/tasks/{task_id}/cancel" in paths


def test_taskcpm_h7b_local_workspace_factory_wires_runtime_boundary(
    _stub_host_llm: None,
) -> None:
    settings = LocalWorkspaceBackendSettings(
        include_task_control=True,
        include_mcp=False,
        include_scheduler=False,
        include_interaction_routes=False,
    )
    app = create_local_workspace_backend_app(
        registry_projection=build_lkw_test_registry_projection(settings),
        settings=settings,
    )
    paths = {route.path for route in app.routes}
    assert "/v1/tasks/{task_id}/cancel" in paths

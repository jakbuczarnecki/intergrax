# © Artur Czarnecki. All rights reserved.

"""TASK-CPM-2 — governed active task autonomy proofs (TASKCPM-A1–A21)."""

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
)
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.harness_task_routes import mount_harness_task_routes
from intergrax.applications._shared.task_control import (
    TaskControlValidationError,
    governed_set_task_autonomy,
)
from intergrax.applications._shared.task_control_governance import (
    MUTATION_TYPE_CANCEL_TASK_EXECUTION,
    MUTATION_TYPE_SET_TASK_AUTONOMY,
    TASK_EXECUTION_RESOURCE_TYPE,
    TaskControlGovernanceBlockedError,
    build_set_task_autonomy_mutation_request,
    task_execution_autonomy_revision,
    task_execution_resource_id,
    task_execution_resource_scope,
)
from intergrax.applications._shared.task_control_wiring import wire_harness_task_control
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
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
from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRequest
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision
from intergrax.contracts.runtime_policy_bundle import (
    PolicyBundleRule,
    build_immutable_runtime_policy_bundle,
)
from intergrax.integrations.contracts.identity_provider import IdentityUser
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.governance.control_plane_mutation_policy import (
    BundleBackedControlPlaneMutationEvaluator,
)
from intergrax.runtime.policy.runtime_policy_bundle_evaluator import RuntimePolicyBundleEvaluator
from intergrax.runtime.task.active_task_registry import ActiveTaskBinding, ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-task-control"
_OTHER_TENANT = "tenant-other"
_MUTATION_ID = "mut-autonomy-1"
_T0 = build_harness_host_control_plane_policy_bundle().issued_at


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


def _task(*, tenant_id: str = _TENANT, autonomy: AutonomyLevel | None = AutonomyLevel.ASK) -> Task:
    task = Task(
        task_id=mint_task_id(),
        tenant_id=tenant_id,
        user_id="user-1",
        message="hello",
        context=TaskContext(),
        state=TaskState.RUNNING,
    )
    task.options.governance.autonomy_level = autonomy
    return task


def _allow_boundary() -> tuple[ControlPlaneMutationAuthorizationBoundary, _RecordingEvaluator]:
    evaluator = _RecordingEvaluator()
    return ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator), evaluator


def _autonomy_request_body(*, run_id: str, level: AutonomyLevel = AutonomyLevel.MANUAL) -> dict[str, str]:
    return {
        "mutation_id": _MUTATION_ID,
        "run_id": run_id,
        "autonomy_level": level.value,
    }


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    ActiveTaskRegistry.clear_for_tests()


@pytest.mark.asyncio
async def test_taskcpm_a1_allow_matching_binding_changes_autonomy_once() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary, evaluator = _allow_boundary()
    result = await governed_set_task_autonomy(
        task_id=str(task.task_id),
        run_id=str(run_id),
        mutation_id=_MUTATION_ID,
        target_autonomy_level=AutonomyLevel.MANUAL,
        principal=_principal(),
        mutation_boundary=boundary,
    )
    assert result.accepted is True
    assert task.options.governance.autonomy_level is AutonomyLevel.MANUAL
    assert len(evaluator.calls) == 1


@pytest.mark.asyncio
async def test_taskcpm_a2_mutation_id_equals_caller_mutation_id() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary, evaluator = _allow_boundary()
    caller_mutation_id = "caller-mut-autonomy"
    await governed_set_task_autonomy(
        task_id=str(task.task_id),
        run_id=str(run_id),
        mutation_id=caller_mutation_id,
        target_autonomy_level=AutonomyLevel.MANUAL,
        principal=_principal(),
        mutation_boundary=boundary,
    )
    assert evaluator.calls[0].mutation_id == caller_mutation_id


@pytest.mark.asyncio
async def test_taskcpm_a3_request_action_is_set_task_autonomy() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary, evaluator = _allow_boundary()
    await governed_set_task_autonomy(
        task_id=str(task.task_id),
        run_id=str(run_id),
        mutation_id=_MUTATION_ID,
        target_autonomy_level=AutonomyLevel.AUTONOMOUS,
        principal=_principal(),
        mutation_boundary=boundary,
    )
    assert evaluator.calls[0].mutation_type == MUTATION_TYPE_SET_TASK_AUTONOMY


@pytest.mark.asyncio
async def test_taskcpm_a4_current_revision_binds_exact_current_autonomy() -> None:
    task = _task(autonomy=AutonomyLevel.ASK)
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary, evaluator = _allow_boundary()
    await governed_set_task_autonomy(
        task_id=str(task.task_id),
        run_id=str(run_id),
        mutation_id=_MUTATION_ID,
        target_autonomy_level=AutonomyLevel.MANUAL,
        principal=_principal(),
        mutation_boundary=boundary,
    )
    assert evaluator.calls[0].current_revision == task_execution_autonomy_revision(
        autonomy_level=AutonomyLevel.ASK,
    )


@pytest.mark.asyncio
async def test_taskcpm_a5_target_revision_binds_requested_autonomy() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary, evaluator = _allow_boundary()
    await governed_set_task_autonomy(
        task_id=str(task.task_id),
        run_id=str(run_id),
        mutation_id=_MUTATION_ID,
        target_autonomy_level=AutonomyLevel.AUTONOMOUS,
        principal=_principal(),
        mutation_boundary=boundary,
    )
    assert evaluator.calls[0].target_revision == task_execution_autonomy_revision(
        autonomy_level=AutonomyLevel.AUTONOMOUS,
    )


@pytest.mark.asyncio
async def test_taskcpm_a6_wrong_tenant_zero_mutation() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary, evaluator = _allow_boundary()
    result = await governed_set_task_autonomy(
        task_id=str(task.task_id),
        run_id=str(run_id),
        mutation_id=_MUTATION_ID,
        target_autonomy_level=AutonomyLevel.MANUAL,
        principal=_principal(tenant_id=_OTHER_TENANT),
        mutation_boundary=boundary,
    )
    assert result.accepted is False
    assert result.detail == "tenant_authority_mismatch"
    assert task.options.governance.autonomy_level is AutonomyLevel.ASK
    assert evaluator.calls == []


@pytest.mark.asyncio
async def test_taskcpm_a7_wrong_run_zero_mutation() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary, evaluator = _allow_boundary()
    result = await governed_set_task_autonomy(
        task_id=str(task.task_id),
        run_id=str(mint_run_id()),
        mutation_id=_MUTATION_ID,
        target_autonomy_level=AutonomyLevel.MANUAL,
        principal=_principal(),
        mutation_boundary=boundary,
    )
    assert result.accepted is False
    assert result.detail == "run_id_mismatch"
    assert task.options.governance.autonomy_level is AutonomyLevel.ASK
    assert evaluator.calls == []


@pytest.mark.asyncio
async def test_taskcpm_a8_deny_zero_mutation_with_evidence() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.DENY,
            reason="deny_autonomy",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="task_control.deny",
            decision_id="dec-deny",
        )
    )
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    result = await governed_set_task_autonomy(
        task_id=str(task.task_id),
        run_id=str(run_id),
        mutation_id=_MUTATION_ID,
        target_autonomy_level=AutonomyLevel.MANUAL,
        principal=_principal(),
        mutation_boundary=boundary,
    )
    assert result.accepted is False
    assert result.blocker_code == "TASK_CONTROL_BLOCKED_BY_POLICY"
    assert task.options.governance.autonomy_level is AutonomyLevel.ASK


@pytest.mark.asyncio
async def test_taskcpm_a9_require_human_zero_mutation_with_scope() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="hitl_required",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="task_control.require_hitl",
            decision_id="dec-hitl",
        )
    )
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    result = await governed_set_task_autonomy(
        task_id=str(task.task_id),
        run_id=str(run_id),
        mutation_id=_MUTATION_ID,
        target_autonomy_level=AutonomyLevel.MANUAL,
        principal=_principal(),
        mutation_boundary=boundary,
    )
    assert result.accepted is False
    assert result.blocker_code == "TASK_CONTROL_BLOCKED_BY_REQUIRE_HUMAN"
    assert result.authorization_scope is not None
    assert task.options.governance.autonomy_level is AutonomyLevel.ASK


@pytest.mark.asyncio
async def test_taskcpm_a10_binding_disappears_after_allow_zero_mutation() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    original_get = ActiveTaskRegistry.get
    lookup_count = 0

    async def _get_then_missing(task_id):  # type: ignore[no-untyped-def]
        nonlocal lookup_count
        lookup_count += 1
        if lookup_count == 1:
            return await original_get(task_id)
        return None

    boundary, _ = _allow_boundary()
    with patch.object(ActiveTaskRegistry, "get", _get_then_missing):
        result = await governed_set_task_autonomy(
            task_id=str(task.task_id),
            run_id=str(run_id),
            mutation_id=_MUTATION_ID,
            target_autonomy_level=AutonomyLevel.MANUAL,
            principal=_principal(),
            mutation_boundary=boundary,
        )
    assert result.accepted is False
    assert result.detail == "stale_active_binding"
    assert task.options.governance.autonomy_level is AutonomyLevel.ASK


@pytest.mark.asyncio
async def test_taskcpm_a11_run_changes_after_allow_zero_mutation() -> None:
    task = _task()
    run_a = mint_run_id()
    run_b = mint_run_id()
    await ActiveTaskRegistry.register(task, run_a)
    replacement = Task(
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        user_id=task.user_id,
        message="replacement",
        state=TaskState.RUNNING,
    )
    replacement.options.governance.autonomy_level = AutonomyLevel.ASK
    replacement_binding = ActiveTaskBinding(
        task_id=task.task_id,
        run_id=run_b,
        task=replacement,
    )
    original_get = ActiveTaskRegistry.get
    lookup_count = 0

    async def _get_with_replacement(task_id):  # type: ignore[no-untyped-def]
        nonlocal lookup_count
        lookup_count += 1
        if lookup_count == 1:
            return await original_get(task_id)
        return replacement_binding

    boundary, _ = _allow_boundary()
    with patch.object(ActiveTaskRegistry, "get", _get_with_replacement):
        result = await governed_set_task_autonomy(
            task_id=str(task.task_id),
            run_id=str(run_a),
            mutation_id=_MUTATION_ID,
            target_autonomy_level=AutonomyLevel.MANUAL,
            principal=_principal(),
            mutation_boundary=boundary,
        )
    assert result.accepted is False
    assert result.detail == "stale_active_binding"
    assert task.options.governance.autonomy_level is AutonomyLevel.ASK


@pytest.mark.asyncio
async def test_taskcpm_a12_current_autonomy_changes_after_allow_zero_mutation() -> None:
    task = _task(autonomy=AutonomyLevel.ASK)
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    changed_task = task.model_copy(deep=True)
    changed_task.options.governance.autonomy_level = AutonomyLevel.MANUAL
    changed_binding = ActiveTaskBinding(task_id=task.task_id, run_id=run_id, task=changed_task)
    original_get = ActiveTaskRegistry.get
    lookup_count = 0

    async def _get_with_changed_autonomy(task_id):  # type: ignore[no-untyped-def]
        nonlocal lookup_count
        lookup_count += 1
        if lookup_count == 1:
            return await original_get(task_id)
        return changed_binding

    boundary, _ = _allow_boundary()
    with patch.object(ActiveTaskRegistry, "get", _get_with_changed_autonomy):
        result = await governed_set_task_autonomy(
            task_id=str(task.task_id),
            run_id=str(run_id),
            mutation_id=_MUTATION_ID,
            target_autonomy_level=AutonomyLevel.AUTONOMOUS,
            principal=_principal(),
            mutation_boundary=boundary,
        )
    assert result.accepted is False
    assert result.detail == "stale_active_binding"
    assert changed_task.options.governance.autonomy_level is AutonomyLevel.MANUAL


@pytest.mark.asyncio
async def test_taskcpm_a13_success_preserves_previous_target_metadata() -> None:
    task = _task(autonomy=AutonomyLevel.ASK)
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary, _ = _allow_boundary()
    result = await governed_set_task_autonomy(
        task_id=str(task.task_id),
        run_id=str(run_id),
        mutation_id=_MUTATION_ID,
        target_autonomy_level=AutonomyLevel.MANUAL,
        principal=_principal(),
        mutation_boundary=boundary,
    )
    assert result.accepted is True
    assert result.metadata == {"previous": AutonomyLevel.ASK.value}
    assert task.options.governance.autonomy_level is AutonomyLevel.MANUAL
    assert task.metadata["autonomy_level"] == AutonomyLevel.MANUAL.value
    assert task.metadata["autonomy_level_previous"] == AutonomyLevel.ASK.value
    assert task.metadata["autonomy_level_changed"] is True


@pytest.mark.asyncio
async def test_taskcpm_a14_same_target_no_op_without_mutation() -> None:
    task = _task(autonomy=AutonomyLevel.MANUAL)
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary, evaluator = _allow_boundary()
    result = await governed_set_task_autonomy(
        task_id=str(task.task_id),
        run_id=str(run_id),
        mutation_id=_MUTATION_ID,
        target_autonomy_level=AutonomyLevel.MANUAL,
        principal=_principal(),
        mutation_boundary=boundary,
    )
    assert result.accepted is True
    assert result.detail == "already_at_target"
    assert evaluator.calls == []
    assert "autonomy_level_changed" not in task.metadata


@pytest.mark.asyncio
async def test_taskcpm_a15_http_route_projects_authenticated_principal() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary, evaluator = _allow_boundary()
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
        mutation_boundary=boundary,
    )
    client = TestClient(app)
    response = client.post(
        f"/v1/tasks/{task.task_id}/autonomy",
        json=_autonomy_request_body(run_id=str(run_id)),
        headers={"Authorization": "Bearer valid-bearer"},
    )
    assert response.status_code == 200
    assert evaluator.calls[0].principal.tenant_id == _TENANT
    assert evaluator.calls[0].principal.user_id == "operator-1"


@pytest.mark.asyncio
async def test_taskcpm_a16_missing_boundary_raises_without_side_effect() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    with pytest.raises(TaskControlGovernanceBlockedError) as exc_info:
        await governed_set_task_autonomy(
            task_id=str(task.task_id),
            run_id=str(run_id),
            mutation_id=_MUTATION_ID,
            target_autonomy_level=AutonomyLevel.MANUAL,
            principal=_principal(),
            mutation_boundary=None,
        )
    assert exc_info.value.blocker_code == "TASK_CONTROL_BLOCKED_BY_MISSING_BOUNDARY"
    assert task.options.governance.autonomy_level is AutonomyLevel.ASK


def test_taskcpm_a17_cancel_policy_rule_does_not_authorize_autonomy() -> None:
    cancel_only_bundle = build_immutable_runtime_policy_bundle(
        bundle_id="harness.control_plane",
        version="1.0.0-test",
        rules=(
            PolicyBundleRule(
                rule_id="harness.task_control.cancel_task_execution",
                match_action=MUTATION_TYPE_CANCEL_TASK_EXECUTION,
                effect="allow",
            ),
        ),
        issued_at=_T0,
    )
    boundary = ControlPlaneMutationAuthorizationBoundary(
        evaluator=BundleBackedControlPlaneMutationEvaluator(
            bundle_evaluator=RuntimePolicyBundleEvaluator(cancel_only_bundle),
        ),
    )
    request = build_set_task_autonomy_mutation_request(
        principal=_principal(),
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        mutation_id=_MUTATION_ID,
        current_autonomy_level=AutonomyLevel.ASK,
        target_autonomy_level=AutonomyLevel.MANUAL,
    )
    result = boundary.authorize(request)
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY


def test_taskcpm_a18_unrelated_mutation_does_not_match_autonomy_rule() -> None:
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


@pytest.mark.asyncio
async def test_taskcpm_a19_product_host_uses_canonical_bundle_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(env: object, agent_override: object | None = None, **_: object) -> object:
        del env
        return agent_override or adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )
    settings = GovernedContractorBackendSettings.from_env()
    manifest = build_governed_contractor_manifest()
    env = manifest.environment or build_governed_contractor_environment_profile(settings)
    runtime = build_harness_host_runtime(
        manifest,
        env,
        registry_projection=build_governed_contractor_test_registry_projection(),
    )
    boundary = resolve_harness_task_control_mutation_boundary(runtime.control_plane_governance)
    assert boundary is not None
    request = build_set_task_autonomy_mutation_request(
        principal=_principal(),
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        mutation_id=_MUTATION_ID,
        current_autonomy_level=AutonomyLevel.ASK,
        target_autonomy_level=AutonomyLevel.MANUAL,
    )
    result = boundary.authorize(request)
    assert result.permitted is True
    assert result.decision.policy_rule_id == "harness.task_control.set_task_autonomy"


@pytest.mark.asyncio
async def test_taskcpm_a20_lab_without_boundary_fail_closed() -> None:
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
    response = client.post(
        f"/v1/tasks/{task.task_id}/autonomy",
        json=_autonomy_request_body(run_id=str(run_id)),
        headers={"Authorization": "Bearer valid-bearer"},
    )
    assert response.status_code == 403
    assert response.json()["detail"]["blocker_code"] == "TASK_CONTROL_BLOCKED_BY_MISSING_BOUNDARY"
    assert task.options.governance.autonomy_level is AutonomyLevel.ASK


def test_taskcpm_a21_http_route_requires_governed_request_shape() -> None:
    task = _task()
    run_id = mint_run_id()
    app = FastAPI()
    app.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(),
        require_api_key=False,
        resolved_api_key=None,
        tenant_required=True,
    )
    boundary, _ = _allow_boundary()
    mount_harness_task_routes(
        app,
        task_runner=UnifiedTaskRunner(object()),  # type: ignore[arg-type]
        mutation_boundary=boundary,
    )
    client = TestClient(app)
    legacy_response = client.post(
        f"/v1/tasks/{task.task_id}/autonomy",
        json={"autonomy_level": AutonomyLevel.MANUAL.value},
        headers={"Authorization": "Bearer valid-bearer"},
    )
    assert legacy_response.status_code == 422


def test_taskcpm_a21b_http_unauthenticated_returns_401() -> None:
    app = FastAPI()
    app.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(),
        require_api_key=False,
        resolved_api_key=None,
        tenant_required=True,
    )
    boundary, _ = _allow_boundary()
    mount_harness_task_routes(
        app,
        task_runner=UnifiedTaskRunner(object()),  # type: ignore[arg-type]
        mutation_boundary=boundary,
    )
    client = TestClient(app)
    response = client.post(
        f"/v1/tasks/{mint_task_id()}/autonomy",
        json=_autonomy_request_body(run_id=str(mint_run_id())),
    )
    assert response.status_code == 401


def test_taskcpm_a4b_unset_current_autonomy_uses_canonical_revision() -> None:
    request = build_set_task_autonomy_mutation_request(
        principal=_principal(),
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        mutation_id=_MUTATION_ID,
        current_autonomy_level=None,
        target_autonomy_level=AutonomyLevel.ASK,
    )
    assert request.current_revision == "autonomy:unset"


@pytest.mark.asyncio
async def test_taskcpm_a10b_authorization_evidence_binds_exact_scope() -> None:
    task = _task(autonomy=AutonomyLevel.ASK)
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary, _ = _allow_boundary()
    result = await governed_set_task_autonomy(
        task_id=str(task.task_id),
        run_id=str(run_id),
        mutation_id=_MUTATION_ID,
        target_autonomy_level=AutonomyLevel.MANUAL,
        principal=_principal(),
        mutation_boundary=boundary,
    )
    evidence = result.authorization_evidence
    assert evidence is not None
    assert evidence.mutation_id == _MUTATION_ID
    assert evidence.mutation_type == MUTATION_TYPE_SET_TASK_AUTONOMY
    assert evidence.resource_type == TASK_EXECUTION_RESOURCE_TYPE
    assert evidence.resource_id == task_execution_resource_id(task_id=task.task_id, run_id=run_id)
    assert evidence.resource_scope == task_execution_resource_scope(
        tenant_id=_TENANT,
        task_id=task.task_id,
        run_id=run_id,
    )
    assert evidence.current_revision == task_execution_autonomy_revision(autonomy_level=AutonomyLevel.ASK)
    assert evidence.target_revision == task_execution_autonomy_revision(autonomy_level=AutonomyLevel.MANUAL)


def test_taskcpm_a16b_lab_profile_has_no_boundary() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id=_TENANT)
    assert env.application_profile is ApplicationProfile.LAB
    governance = build_harness_control_plane_governance(env)
    assert resolve_harness_task_control_mutation_boundary(governance) is None


@pytest.mark.asyncio
async def test_taskcpm_a16c_missing_mutation_id_fails_before_side_effect() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary, evaluator = _allow_boundary()
    with pytest.raises(TaskControlValidationError, match="mutation_id_required"):
        await governed_set_task_autonomy(
            task_id=str(task.task_id),
            run_id=str(run_id),
            mutation_id="   ",
            target_autonomy_level=AutonomyLevel.MANUAL,
            principal=_principal(),
            mutation_boundary=boundary,
        )
    assert evaluator.calls == []
    assert task.options.governance.autonomy_level is AutonomyLevel.ASK

# © Artur Czarnecki. All rights reserved.

"""TASK-CPM-4 — supported task-control mutation bypass closure (TASKCPM-B1–B15)."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from governed_contractor_application.host.factory import create_governed_contractor_backend_app
from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.manifest import build_governed_contractor_manifest
from governed_contractor_application.tests.governed_contractor_ac3_projection import (
    build_governed_contractor_test_registry_projection,
)
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
    _execute_autonomy_change,
    governed_cancel_active_task,
    governed_resume_checkpoint_task,
    governed_set_task_autonomy,
)
from intergrax.applications._shared.task_control_governance import (
    MUTATION_TYPE_CANCEL_TASK_EXECUTION,
    build_cancel_task_execution_mutation_request,
    build_resume_task_execution_mutation_request,
    build_set_task_autonomy_mutation_request,
)
from intergrax.applications._shared.task_control_wiring import wire_harness_task_control
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_decision import HumanRequest
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRequest
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id, mint_task_id
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision
from intergrax.contracts.runtime_policy_bundle import (
    PolicyBundleRule,
    build_immutable_runtime_policy_bundle,
)
from intergrax.integrations.contracts.identity_provider import IdentityUser
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.governance.control_plane_mutation_policy import (
    BundleBackedControlPlaneMutationEvaluator,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.execution_tree_checkpoint import minimal_runtime_checkpoint
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
from intergrax.runtime.policy.runtime_policy_bundle_evaluator import RuntimePolicyBundleEvaluator
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from intergrax.runtime.task.task_contract import TaskPauseRecord
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-bypass-closure"
_MUTATION_ID = "mut-bypass-1"
_TASK_ID = mint_task_id()
_RUN_ID = mint_run_id()
_ATTEMPT_ID = mint_attempt_id()
_ROOT_EXECUTION_ID = mint_execution_id()
_PAUSE_ID = "pause-bypass"
_HUMAN_REQUEST_ID = "hr-bypass"
_RESUME_TOKEN = "resume-token-bypass"
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


def _task(*, tenant_id: str = _TENANT, autonomy: AutonomyLevel = AutonomyLevel.ASK) -> Task:
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
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    return boundary, evaluator


def _deny_boundary() -> ControlPlaneMutationAuthorizationBoundary:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.DENY,
            reason="test_deny",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="task_control.deny",
            decision_id="dec-deny",
        )
    )
    return ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)


def _paused_task(*, tenant_id: str = _TENANT) -> Task:
    task = Task(
        tenant_id=tenant_id,
        user_id="task-owner",
        message="paused work",
        task_id=_TASK_ID,
    )
    task.runtime.governance.paused = True
    task.runtime.governance.pause_record = TaskPauseRecord(
        pause_id=_PAUSE_ID,
        task_id=_TASK_ID,
        human_request_id=_HUMAN_REQUEST_ID,
    )
    task.runtime.governance.human_request = HumanRequest(
        request_id=_HUMAN_REQUEST_ID,
        prompt="approve?",
    )
    return task


def _checkpoint(*, tenant_id: str = _TENANT) -> TaskCheckpoint:
    task = _paused_task(tenant_id=tenant_id)
    return TaskCheckpoint(
        checkpoint_id="cp-bypass-1",
        task_id=str(_TASK_ID),
        tenant_id=tenant_id,
        resume_token=_RESUME_TOKEN,
        task_snapshot=task.model_dump(mode="json"),
        task_state=TaskState.WAITING_FOR_HUMAN,
        progress_message="paused",
        notify_channel="debug",
        runtime=minimal_runtime_checkpoint(
            task_id=_TASK_ID,
            run_id=_RUN_ID,
            attempt_id=_ATTEMPT_ID,
            root_execution_id=_ROOT_EXECUTION_ID,
        ),
    )


class _StaticCheckpointStore:
    def __init__(self, checkpoint: TaskCheckpoint) -> None:
        self._checkpoint = checkpoint

    def get_by_token(self, task_id: str, tenant_id: str, resume_token: str) -> TaskCheckpoint | None:
        if (
            task_id == self._checkpoint.task_id
            and tenant_id == self._checkpoint.tenant_id
            and resume_token == self._checkpoint.resume_token
        ):
            return self._checkpoint
        return None

    def get_latest(self, task_id: str, tenant_id: str) -> TaskCheckpoint | None:
        del task_id, tenant_id
        return None


def _product_runtime() -> object:
    settings = GovernedContractorBackendSettings.from_env()
    manifest = build_governed_contractor_manifest()
    env = manifest.environment or build_governed_contractor_environment_profile(settings)
    return build_harness_host_runtime(
        manifest,
        env,
        registry_projection=build_governed_contractor_test_registry_projection(),
    )


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    ActiveTaskRegistry.clear_for_tests()


@pytest.fixture
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(env: object, agent_override: object | None = None, **_: object) -> object:
        del env
        return agent_override or adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )


@pytest.mark.asyncio
async def test_taskcpm_b1_supported_cancel_route_reaches_coordinator_only_after_allow(
    _stub_host_llm: None,
) -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    app_deny = FastAPI()
    app_deny.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(),
        require_api_key=False,
        resolved_api_key=None,
        tenant_required=True,
    )
    mount_harness_task_routes(
        app_deny,
        task_runner=UnifiedTaskRunner(object()),  # type: ignore[arg-type]
        mutation_boundary=_deny_boundary(),
    )
    client_deny = TestClient(app_deny)
    with patch(
        "intergrax.applications._shared.task_control.CancellationCoordinator.request",
    ) as cancel_request_deny:
        deny_response = client_deny.post(
            f"/v1/tasks/{task.task_id}/cancel",
            json={"mutation_id": _MUTATION_ID, "run_id": str(run_id), "reason": "deny_probe"},
            headers={"Authorization": "Bearer valid-bearer"},
        )
    assert deny_response.status_code == 403
    assert cancel_request_deny.call_count == 0

    runtime_allow = _product_runtime()
    task_allow = _task()
    run_id_allow = mint_run_id()
    await ActiveTaskRegistry.register(task_allow, run_id_allow)
    app_allow = FastAPI()
    app_allow.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(),
        require_api_key=False,
        resolved_api_key=None,
        tenant_required=True,
    )
    wire_harness_task_control(
        app_allow,
        enabled=True,
        task_runner=UnifiedTaskRunner(runtime_allow.nexus_loop),  # type: ignore[arg-type]
        env=runtime_allow.environment,
        runtime=runtime_allow,
    )
    client_allow = TestClient(app_allow)
    with patch(
        "intergrax.applications._shared.task_control.CancellationCoordinator.request",
        wraps=CancellationCoordinator.request,
    ) as cancel_request_allow:
        allow_response = client_allow.post(
            f"/v1/tasks/{task_allow.task_id}/cancel",
            json={"mutation_id": _MUTATION_ID, "run_id": str(run_id_allow)},
            headers={"Authorization": "Bearer valid-bearer"},
        )
    assert allow_response.status_code == 200
    assert cancel_request_allow.call_count == 1


def test_taskcpm_b2_supported_cancel_surface_exports_governed_facade() -> None:
    assert callable(governed_cancel_active_task)


@pytest.mark.asyncio
async def test_taskcpm_b4_supported_autonomy_route_reaches_mutation_only_through_governed_facade() -> None:
    task_deny = _task()
    run_id_deny = mint_run_id()
    await ActiveTaskRegistry.register(task_deny, run_id_deny)
    app_deny = FastAPI()
    app_deny.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(),
        require_api_key=False,
        resolved_api_key=None,
        tenant_required=True,
    )
    mount_harness_task_routes(
        app_deny,
        task_runner=UnifiedTaskRunner(object()),  # type: ignore[arg-type]
        mutation_boundary=_deny_boundary(),
    )
    client_deny = TestClient(app_deny)
    original_level = task_deny.options.governance.autonomy_level
    with patch(
        "intergrax.applications._shared.task_control._execute_autonomy_change",
    ) as autonomy_change_deny:
        deny_response = client_deny.post(
            f"/v1/tasks/{task_deny.task_id}/autonomy",
            json={
                "mutation_id": _MUTATION_ID,
                "run_id": str(run_id_deny),
                "autonomy_level": "manual",
            },
            headers={"Authorization": "Bearer valid-bearer"},
        )
    assert deny_response.status_code == 403
    assert autonomy_change_deny.call_count == 0
    assert task_deny.options.governance.autonomy_level is original_level

    task_allow = _task()
    run_id_allow = mint_run_id()
    await ActiveTaskRegistry.register(task_allow, run_id_allow)
    boundary, evaluator = _allow_boundary()
    app_allow = FastAPI()
    app_allow.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(),
        require_api_key=False,
        resolved_api_key=None,
        tenant_required=True,
    )
    mount_harness_task_routes(
        app_allow,
        task_runner=UnifiedTaskRunner(object()),  # type: ignore[arg-type]
        mutation_boundary=boundary,
    )
    client_allow = TestClient(app_allow)
    with patch(
        "intergrax.applications._shared.task_control._execute_autonomy_change",
        wraps=_execute_autonomy_change,
    ) as autonomy_change_allow:
        allow_response = client_allow.post(
            f"/v1/tasks/{task_allow.task_id}/autonomy",
            json={
                "mutation_id": _MUTATION_ID,
                "run_id": str(run_id_allow),
                "autonomy_level": "manual",
            },
            headers={"Authorization": "Bearer valid-bearer"},
        )
    assert allow_response.status_code == 200
    assert len(evaluator.calls) == 1
    assert autonomy_change_allow.call_count == 1


@pytest.mark.asyncio
async def test_taskcpm_b6_supported_operator_resume_reaches_runner_through_governed_facade() -> None:
    checkpoint = _checkpoint()
    boundary, evaluator = _allow_boundary()
    runner = AsyncMock(spec=UnifiedTaskRunner)
    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
        return_value=TaskResult(task_id=_TASK_ID, state=TaskState.COMPLETED, answer="ok"),
    ) as resume_call:
        outcome = await governed_resume_checkpoint_task(
            runner,
            task_id=_TASK_ID,
            tenant_id=_TENANT,
            resume_token=_RESUME_TOKEN,
            mutation_id=_MUTATION_ID,
            principal=_principal(),
            mutation_boundary=boundary,
            checkpoint_store=_StaticCheckpointStore(checkpoint),
            operator_input={"verdict": "approve"},
        )
    assert outcome.accepted is True
    assert len(evaluator.calls) == 1
    assert resume_call.await_count == 1


@pytest.mark.asyncio
async def test_taskcpm_b7_supported_resume_route_reaches_runner_only_through_governed_facade() -> None:
    checkpoint = _checkpoint()
    app_deny = FastAPI()
    app_deny.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(),
        require_api_key=False,
        resolved_api_key=None,
        tenant_required=True,
    )
    runner_deny = AsyncMock(spec=UnifiedTaskRunner)
    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
    ) as resume_deny:
        mount_harness_task_routes(
            app_deny,
            task_runner=runner_deny,
            checkpoint_store=_StaticCheckpointStore(checkpoint),
            mutation_boundary=_deny_boundary(),
        )
        client_deny = TestClient(app_deny)
        deny_response = client_deny.post(
            f"/v1/tasks/{_TASK_ID}/resume",
            json={
                "mutation_id": _MUTATION_ID,
                "resume_token": _RESUME_TOKEN,
                "operator_input": {"verdict": "approve"},
            },
            headers={"Authorization": "Bearer valid-bearer"},
        )
    assert deny_response.status_code == 403
    assert resume_deny.await_count == 0

    app_allow = FastAPI()
    app_allow.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(),
        require_api_key=False,
        resolved_api_key=None,
        tenant_required=True,
    )
    boundary, evaluator = _allow_boundary()
    runner_allow = AsyncMock(spec=UnifiedTaskRunner)
    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
        return_value=TaskResult(task_id=_TASK_ID, state=TaskState.COMPLETED, answer="ok"),
    ) as resume_allow:
        mount_harness_task_routes(
            app_allow,
            task_runner=runner_allow,
            checkpoint_store=_StaticCheckpointStore(checkpoint),
            mutation_boundary=boundary,
        )
        client_allow = TestClient(app_allow)
        allow_response = client_allow.post(
            f"/v1/tasks/{_TASK_ID}/resume",
            json={
                "mutation_id": _MUTATION_ID,
                "resume_token": _RESUME_TOKEN,
                "operator_input": {"verdict": "approve"},
            },
            headers={"Authorization": "Bearer valid-bearer"},
        )
    assert allow_response.status_code == 200
    assert len(evaluator.calls) == 1
    assert resume_allow.await_count == 1


@pytest.mark.asyncio
async def test_taskcpm_b8_debug_hitl_resume_service_is_debug_lab_only() -> None:
    from intergrax.debug.hitl_service import DebugHitlResumeService
    from intergrax.runtime.human.models import HumanResponseVerdict
    from intergrax.runtime.registry.agent_registry import AgentRegistry

    checkpoint = _checkpoint()
    service = DebugHitlResumeService(
        AgentRegistry(),
        checkpoint_store=_StaticCheckpointStore(checkpoint),
    )
    with patch(
        "intergrax.applications._shared.task_control.governed_resume_checkpoint_task",
        new_callable=AsyncMock,
    ) as governed_resume:
        with patch(
            "intergrax.debug.hitl_service.NexusLoop.handle_task",
            new_callable=AsyncMock,
            return_value=TaskResult(task_id=_TASK_ID, state=TaskState.COMPLETED, answer="ok"),
        ) as handle_task:
            await service.resume_with_human_response(
                str(_TASK_ID),
                _TENANT,
                verdict=HumanResponseVerdict.APPROVE,
                resume_token=_RESUME_TOKEN,
            )
    governed_resume.assert_not_awaited()
    handle_task.assert_awaited_once()


@pytest.mark.asyncio
async def test_taskcpm_b9_supported_cancel_route_does_not_invoke_taskqueue_cancel(
    _stub_host_llm: None,
) -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    runtime = _product_runtime()
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
        "intergrax.queueing.contracts.task_queue.TaskQueue.cancel",
        new_callable=MagicMock,
    ) as task_queue_cancel:
        response = client.post(
            f"/v1/tasks/{task.task_id}/cancel",
            json={"mutation_id": _MUTATION_ID, "run_id": str(run_id)},
            headers={"Authorization": "Bearer valid-bearer"},
        )
    assert response.status_code == 200
    task_queue_cancel.assert_not_called()


def test_taskcpm_b10_product_scaffold_wiring_uses_shared_governed_task_control(
    _stub_host_llm: None,
) -> None:
    runtime = _product_runtime()
    app = FastAPI()
    wire_harness_task_control(
        app,
        enabled=True,
        task_runner=UnifiedTaskRunner(runtime.nexus_loop),  # type: ignore[arg-type]
        env=runtime.environment,
        runtime=runtime,
    )
    paths = {route.path for route in app.routes}
    assert "/v1/tasks/{task_id}/cancel" in paths
    assert "/v1/tasks/{task_id}/autonomy" in paths
    assert "/v1/tasks/{task_id}/resume" in paths


def test_taskcpm_b11_product_host_exposes_no_duplicate_raw_task_control_routes(
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
    paths = sorted({route.path for route in app.routes if "{task_id}" in route.path})
    assert paths.count("/v1/tasks/{task_id}/cancel") == 1
    assert paths.count("/v1/tasks/{task_id}/autonomy") == 1
    assert paths.count("/v1/tasks/{task_id}/resume") == 1
    assert not any("/debug/" in path and path.endswith("/cancel") for path in paths)


def test_taskcpm_b12_lab_fail_closed_boundary_behavior_unchanged() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id=_TENANT)
    assert env.application_profile is ApplicationProfile.LAB
    governance = build_harness_control_plane_governance(env)
    assert resolve_harness_task_control_mutation_boundary(governance) is None


@pytest.mark.asyncio
async def test_taskcpm_b12b_lab_direct_mount_cancel_remains_fail_closed() -> None:
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
    assert cancel_request.call_count == 0


def test_taskcpm_b13_cancel_autonomy_resume_bundle_rules_remain_isolated() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_TENANT)
    boundary = resolve_harness_task_control_mutation_boundary(
        build_harness_control_plane_governance(env),
    )
    assert boundary is not None
    cancel_request = build_cancel_task_execution_mutation_request(
        principal=_principal(),
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        mutation_id=_MUTATION_ID,
        current_state=TaskState.RUNNING,
    )
    autonomy_request = build_set_task_autonomy_mutation_request(
        principal=_principal(),
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        mutation_id=_MUTATION_ID,
        current_autonomy_level=AutonomyLevel.ASK,
        target_autonomy_level=AutonomyLevel.MANUAL,
    )
    resume_request = build_resume_task_execution_mutation_request(
        principal=_principal(),
        tenant_id=_TENANT,
        task_id=_TASK_ID,
        run_id=_RUN_ID,
        mutation_id=_MUTATION_ID,
        checkpoint=_checkpoint(),
    )
    cancel_only = ControlPlaneMutationAuthorizationBoundary(
        evaluator=BundleBackedControlPlaneMutationEvaluator(
            bundle_evaluator=RuntimePolicyBundleEvaluator(
                build_immutable_runtime_policy_bundle(
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
            ),
        ),
    )
    assert cancel_only.authorize(cancel_request).permitted is True
    assert cancel_only.authorize(autonomy_request).permitted is False
    assert cancel_only.authorize(resume_request).permitted is False
    assert boundary.authorize(cancel_request).decision.policy_rule_id.endswith("cancel_task_execution")
    assert boundary.authorize(autonomy_request).decision.policy_rule_id.endswith("set_task_autonomy")
    assert boundary.authorize(resume_request).decision.policy_rule_id.endswith("resume_task_execution")


def test_taskcpm_b14_scheduler_resume_stays_on_runtime_internal_path() -> None:
    from intergrax.runtime.long_running.scheduler import LongRunningScheduler

    assert governed_resume_checkpoint_task.__module__.endswith("task_control")
    assert LongRunningScheduler.__module__.endswith("scheduler")


@pytest.mark.asyncio
async def test_taskcpm_b14b_governed_cancel_still_reaches_internal_cooperative_cancel() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary, _ = _allow_boundary()
    with patch(
        "intergrax.applications._shared.task_control.CancellationCoordinator.request",
        wraps=CancellationCoordinator.request,
    ) as cancel_request:
        result = await governed_cancel_active_task(
            task_id=str(task.task_id),
            run_id=str(run_id),
            mutation_id=_MUTATION_ID,
            principal=_principal(),
            mutation_boundary=boundary,
        )
    assert result.accepted is True
    assert cancel_request.call_count == 1


@pytest.mark.asyncio
async def test_taskcpm_b15_governed_autonomy_regression_still_passes_shape() -> None:
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
    assert len(evaluator.calls) == 1
    assert task.options.governance.autonomy_level is AutonomyLevel.MANUAL

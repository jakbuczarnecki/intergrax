# © Artur Czarnecki. All rights reserved.

"""TASK-CPM-1 — governed cooperative task cancel proofs (TASKCPM-C1–C15)."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.applications._shared.harness_task_routes import mount_harness_task_routes
from intergrax.applications._shared.task_control import (
    TaskControlValidationError,
    governed_cancel_active_task,
)
from intergrax.applications._shared.task_control_governance import (
    MUTATION_TYPE_CANCEL_TASK_EXECUTION,
    TASK_EXECUTION_RESOURCE_TYPE,
    TaskControlGovernanceBlockedError,
    build_cancel_task_execution_mutation_request,
    task_execution_resource_id,
    task_execution_resource_scope,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRequest
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision
from intergrax.runtime.cancellation.coordinator import (
    CANCELLATION_REASON_KEY,
    CancellationCoordinator,
)
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.task.active_task_registry import ActiveTaskBinding, ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.integrations.contracts.identity_provider import IdentityUser
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-task-control"
_OTHER_TENANT = "tenant-other"
_MUTATION_ID = "mut-cancel-1"
_REASON = "operator_requested"


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
    on_evaluate: object | None = None

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        self.calls.append(request)
        if self.on_evaluate is not None:
            self.on_evaluate()
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


def _allow_boundary(
    *,
    on_evaluate: object | None = None,
) -> tuple[ControlPlaneMutationAuthorizationBoundary, _RecordingEvaluator]:
    evaluator = _RecordingEvaluator(on_evaluate=on_evaluate)
    return ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator), evaluator


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    ActiveTaskRegistry.clear_for_tests()


@pytest.mark.asyncio
async def test_taskcpm_c1_allow_matching_binding_requests_cancel_once() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary, evaluator = _allow_boundary()
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
            reason=_REASON,
        )
    assert result.accepted is True
    assert cancel_request.call_count == 1
    assert cancel_request.call_args.kwargs["reason"] == _REASON
    assert len(evaluator.calls) == 1
    assert CancellationCoordinator.is_requested(task.metadata)


@pytest.mark.asyncio
async def test_taskcpm_c2_deny_zero_cancellation_effect() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.DENY,
            reason="deny_cancel",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="task_control.deny",
            decision_id="dec-deny",
        )
    )
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    with patch(
        "intergrax.applications._shared.task_control.CancellationCoordinator.request",
    ) as cancel_request:
        result = await governed_cancel_active_task(
            task_id=str(task.task_id),
            run_id=str(run_id),
            mutation_id=_MUTATION_ID,
            principal=_principal(),
            mutation_boundary=boundary,
        )
    assert result.accepted is False
    assert result.blocker_code == "TASK_CONTROL_BLOCKED_BY_POLICY"
    assert cancel_request.call_count == 0
    assert not CancellationCoordinator.is_requested(task.metadata)


@pytest.mark.asyncio
async def test_taskcpm_c3_require_human_zero_cancellation_effect() -> None:
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
    with patch(
        "intergrax.applications._shared.task_control.CancellationCoordinator.request",
    ) as cancel_request:
        result = await governed_cancel_active_task(
            task_id=str(task.task_id),
            run_id=str(run_id),
            mutation_id=_MUTATION_ID,
            principal=_principal(),
            mutation_boundary=boundary,
        )
    assert result.accepted is False
    assert result.blocker_code == "TASK_CONTROL_BLOCKED_BY_REQUIRE_HUMAN"
    assert result.authorization_scope is not None
    assert cancel_request.call_count == 0


@pytest.mark.asyncio
async def test_taskcpm_c4_wrong_tenant_zero_cancellation_effect() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary, _ = _allow_boundary()
    with patch(
        "intergrax.applications._shared.task_control.CancellationCoordinator.request",
    ) as cancel_request:
        result = await governed_cancel_active_task(
            task_id=str(task.task_id),
            run_id=str(run_id),
            mutation_id=_MUTATION_ID,
            principal=_principal(tenant_id=_OTHER_TENANT),
            mutation_boundary=boundary,
        )
    assert result.accepted is False
    assert result.detail == "tenant_authority_mismatch"
    assert cancel_request.call_count == 0
    assert boundary.evaluator.calls == []


@pytest.mark.asyncio
async def test_taskcpm_c5_wrong_run_id_zero_cancellation_effect() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary, evaluator = _allow_boundary()
    with patch(
        "intergrax.applications._shared.task_control.CancellationCoordinator.request",
    ) as cancel_request:
        result = await governed_cancel_active_task(
            task_id=str(task.task_id),
            run_id=str(mint_run_id()),
            mutation_id=_MUTATION_ID,
            principal=_principal(),
            mutation_boundary=boundary,
        )
    assert result.accepted is False
    assert result.detail == "run_id_mismatch"
    assert cancel_request.call_count == 0
    assert evaluator.calls == []


@pytest.mark.asyncio
async def test_taskcpm_c6_binding_changes_after_authorization_zero_effect() -> None:
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
        with patch(
            "intergrax.applications._shared.task_control.CancellationCoordinator.request",
        ) as cancel_request:
            result = await governed_cancel_active_task(
                task_id=str(task.task_id),
                run_id=str(run_a),
                mutation_id=_MUTATION_ID,
                principal=_principal(),
                mutation_boundary=boundary,
            )
    assert result.accepted is False
    assert result.detail == "stale_active_binding"
    assert cancel_request.call_count == 0
    assert not CancellationCoordinator.is_requested(replacement.metadata)


@pytest.mark.asyncio
async def test_taskcpm_c7_task_disappears_after_authorization_zero_effect() -> None:
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
        with patch(
            "intergrax.applications._shared.task_control.CancellationCoordinator.request",
        ) as cancel_request:
            result = await governed_cancel_active_task(
                task_id=str(task.task_id),
                run_id=str(run_id),
                mutation_id=_MUTATION_ID,
                principal=_principal(),
                mutation_boundary=boundary,
            )
    assert result.accepted is False
    assert result.detail == "stale_active_binding"
    assert cancel_request.call_count == 0


@pytest.mark.asyncio
async def test_taskcpm_c8_terminal_state_after_authorization_zero_effect() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    completed_task = task.model_copy(update={"state": TaskState.COMPLETED})
    completed_binding = ActiveTaskBinding(
        task_id=task.task_id,
        run_id=run_id,
        task=completed_task,
    )
    original_get = ActiveTaskRegistry.get
    lookup_count = 0

    async def _get_then_terminal(task_id):  # type: ignore[no-untyped-def]
        nonlocal lookup_count
        lookup_count += 1
        if lookup_count == 1:
            return await original_get(task_id)
        return completed_binding

    boundary, _ = _allow_boundary()
    with patch.object(ActiveTaskRegistry, "get", _get_then_terminal):
        with patch(
            "intergrax.applications._shared.task_control.CancellationCoordinator.request",
        ) as cancel_request:
            result = await governed_cancel_active_task(
                task_id=str(task.task_id),
                run_id=str(run_id),
                mutation_id=_MUTATION_ID,
                principal=_principal(),
                mutation_boundary=boundary,
            )
    assert result.accepted is False
    assert result.detail == "stale_active_binding"
    assert cancel_request.call_count == 0


@pytest.mark.asyncio
async def test_taskcpm_c9_mutation_id_equals_caller_mutation_id() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary, evaluator = _allow_boundary()
    caller_mutation_id = "caller-mut-abc"
    await governed_cancel_active_task(
        task_id=str(task.task_id),
        run_id=str(run_id),
        mutation_id=caller_mutation_id,
        principal=_principal(),
        mutation_boundary=boundary,
    )
    assert len(evaluator.calls) == 1
    assert evaluator.calls[0].mutation_id == caller_mutation_id


@pytest.mark.asyncio
async def test_taskcpm_c10_authorization_evidence_binds_exact_scope() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary, _ = _allow_boundary()
    result = await governed_cancel_active_task(
        task_id=str(task.task_id),
        run_id=str(run_id),
        mutation_id=_MUTATION_ID,
        principal=_principal(),
        mutation_boundary=boundary,
        reason=_REASON,
    )
    evidence = result.authorization_evidence
    assert evidence is not None
    assert evidence.mutation_id == _MUTATION_ID
    assert evidence.mutation_type == MUTATION_TYPE_CANCEL_TASK_EXECUTION
    assert evidence.tenant_id == _TENANT
    assert evidence.task_id == str(task.task_id)
    assert evidence.run_id == str(run_id)
    assert evidence.resource_type == TASK_EXECUTION_RESOURCE_TYPE
    assert evidence.resource_id == task_execution_resource_id(task_id=task.task_id, run_id=run_id)
    assert evidence.resource_scope == task_execution_resource_scope(
        tenant_id=_TENANT,
        task_id=task.task_id,
        run_id=run_id,
    )
    assert evidence.current_revision == f"state:{TaskState.RUNNING.value}"
    assert evidence.target_revision == "state:cancel_requested"


@pytest.mark.asyncio
async def test_taskcpm_c11_http_route_projects_authenticated_principal() -> None:
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
        f"/v1/tasks/{task.task_id}/cancel",
        json={
            "mutation_id": _MUTATION_ID,
            "run_id": str(run_id),
            "reason": _REASON,
        },
        headers={"Authorization": "Bearer valid-bearer"},
    )
    assert response.status_code == 200
    assert len(evaluator.calls) == 1
    assert evaluator.calls[0].principal.tenant_id == _TENANT
    assert evaluator.calls[0].principal.user_id == "operator-1"


@pytest.mark.asyncio
async def test_taskcpm_c12_missing_mutation_id_fails_before_side_effect() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary, evaluator = _allow_boundary()
    with patch(
        "intergrax.applications._shared.task_control.CancellationCoordinator.request",
    ) as cancel_request:
        with pytest.raises(TaskControlValidationError, match="mutation_id_required"):
            await governed_cancel_active_task(
                task_id=str(task.task_id),
                run_id=str(run_id),
                mutation_id="   ",
                principal=_principal(),
                mutation_boundary=boundary,
            )
    assert cancel_request.call_count == 0
    assert evaluator.calls == []


@pytest.mark.asyncio
async def test_taskcpm_c13_reason_preserved_to_coordinator() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary, _ = _allow_boundary()
    custom_reason = "maintenance_window"
    with patch(
        "intergrax.applications._shared.task_control.CancellationCoordinator.request",
        wraps=CancellationCoordinator.request,
    ) as cancel_request:
        await governed_cancel_active_task(
            task_id=str(task.task_id),
            run_id=str(run_id),
            mutation_id=_MUTATION_ID,
            principal=_principal(),
            mutation_boundary=boundary,
            reason=custom_reason,
        )
    assert cancel_request.call_args.kwargs["reason"] == custom_reason


@pytest.mark.asyncio
async def test_taskcpm_c14_cooperative_cancel_behavior_preserved_after_allow() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary, _ = _allow_boundary()
    result = await governed_cancel_active_task(
        task_id=str(task.task_id),
        run_id=str(run_id),
        mutation_id=_MUTATION_ID,
        principal=_principal(),
        mutation_boundary=boundary,
        reason=_REASON,
    )
    assert result.accepted is True
    assert CancellationCoordinator.is_requested(task.metadata)
    assert task.metadata[CANCELLATION_REASON_KEY] == _REASON


@pytest.mark.asyncio
async def test_taskcpm_c15_missing_boundary_raises_without_side_effect() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    with patch(
        "intergrax.applications._shared.task_control.CancellationCoordinator.request",
    ) as cancel_request:
        with pytest.raises(TaskControlGovernanceBlockedError) as exc_info:
            await governed_cancel_active_task(
                task_id=str(task.task_id),
                run_id=str(run_id),
                mutation_id=_MUTATION_ID,
                principal=_principal(),
                mutation_boundary=None,
            )
    assert exc_info.value.blocker_code == "TASK_CONTROL_BLOCKED_BY_MISSING_BOUNDARY"
    assert cancel_request.call_count == 0


def test_taskcpm_c15b_build_mutation_request_binds_task_and_run() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    request = build_cancel_task_execution_mutation_request(
        principal=_principal(),
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        mutation_id=_MUTATION_ID,
        current_state=TaskState.RUNNING,
    )
    assert request.task_id == task_id
    assert request.run_id == run_id
    assert request.mutation_type == MUTATION_TYPE_CANCEL_TASK_EXECUTION

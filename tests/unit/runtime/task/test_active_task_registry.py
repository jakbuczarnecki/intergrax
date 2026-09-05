# © Artur Czarnecki. All rights reserved.

"""TASK-CPM-0 — active execution binding for ActiveTaskRegistry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.task_control import (
    governed_cancel_active_task,
    governed_set_task_autonomy,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.runtime_policy import (
    EnforcementLevel,
    PolicyAction,
    PolicyDecision,
)
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.contracts.active_execution_task_scope import (
    ActiveExecutionTaskScopeUnavailable,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.task.active_task_registry import (
    ActiveRunOwnershipConflict,
    ActiveTaskBinding,
    ActiveTaskOwnershipConflict,
    ActiveTaskRegistry,
    ActiveTaskRegistryTaskScopeResolver,
)
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _task() -> Task:
    return Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="hello",
        context=TaskContext(),
    )


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    ActiveTaskRegistry.clear_for_tests()


@pytest.mark.asyncio
async def test_taskreg_1_register_first_task_run_succeeds() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    binding = await ActiveTaskRegistry.get(task.task_id)
    assert binding is not None


@pytest.mark.asyncio
async def test_taskreg_2_lookup_returns_exact_task_and_run_id() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    binding = await ActiveTaskRegistry.get(task.task_id)
    assert binding is not None
    assert isinstance(binding, ActiveTaskBinding)
    assert binding.task is task
    assert binding.run_id == run_id
    assert binding.task_id == task.task_id


@pytest.mark.asyncio
async def test_taskreg_3_same_task_id_different_run_id_cannot_overwrite() -> None:
    task = _task()
    run_a = mint_run_id()
    run_b = mint_run_id()
    await ActiveTaskRegistry.register(task, run_a)
    other = Task(
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        user_id=task.user_id,
        message="other",
    )
    with pytest.raises(ActiveTaskOwnershipConflict) as exc_info:
        await ActiveTaskRegistry.register(other, run_b)
    conflict = exc_info.value
    assert conflict.existing_run_id == run_a
    assert conflict.requested_run_id == run_b


@pytest.mark.asyncio
async def test_taskreg_4_existing_binding_remains_after_rejected_overwrite() -> None:
    task = _task()
    run_a = mint_run_id()
    run_b = mint_run_id()
    await ActiveTaskRegistry.register(task, run_a)
    other = Task(
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        user_id=task.user_id,
        message="other",
    )
    with pytest.raises(ActiveTaskOwnershipConflict):
        await ActiveTaskRegistry.register(other, run_b)
    binding = await ActiveTaskRegistry.get(task.task_id)
    assert binding is not None
    assert binding.task is task
    assert binding.run_id == run_a
    assert ActiveTaskRegistry.peek_task_id_for_run(run_a) == task.task_id
    assert ActiveTaskRegistry.peek_task_id_for_run(run_b) is None


@pytest.mark.asyncio
async def test_taskreg_5_same_run_re_register_is_idempotent() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    refreshed = Task(
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        user_id=task.user_id,
        message="refreshed",
    )
    await ActiveTaskRegistry.register(refreshed, run_id)
    binding = await ActiveTaskRegistry.get(task.task_id)
    assert binding is not None
    assert binding.task is refreshed
    assert binding.run_id == run_id
    assert ActiveTaskRegistry.peek_task_id_for_run(run_id) == task.task_id


@pytest.mark.asyncio
async def test_taskreg_5b_same_run_different_task_raises_run_ownership_conflict() -> (
    None
):
    task_a = _task()
    task_b = _task()
    run_r = mint_run_id()
    await ActiveTaskRegistry.register(task_a, run_r)
    with pytest.raises(ActiveRunOwnershipConflict) as exc_info:
        await ActiveTaskRegistry.register(task_b, run_r)
    conflict = exc_info.value
    assert conflict.run_id == run_r
    assert conflict.existing_task_id == task_a.task_id
    assert conflict.requested_task_id == task_b.task_id
    binding_a = await ActiveTaskRegistry.get(task_a.task_id)
    assert binding_a is not None
    assert binding_a.run_id == run_r
    assert await ActiveTaskRegistry.get(task_b.task_id) is None
    assert ActiveTaskRegistry.peek_task_id_for_run(run_r) == task_a.task_id


@pytest.mark.asyncio
async def test_taskreg_6_unregister_exact_task_id_run_id_removes_binding() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    removed = await ActiveTaskRegistry.unregister(task.task_id, run_id)
    assert removed is True
    assert await ActiveTaskRegistry.get(task.task_id) is None
    assert ActiveTaskRegistry.peek_task_id_for_run(run_id) is None


@pytest.mark.asyncio
async def test_taskreg_7_wrong_run_id_cannot_unregister_current_binding() -> None:
    task = _task()
    run_current = mint_run_id()
    run_stale = mint_run_id()
    await ActiveTaskRegistry.register(task, run_current)
    removed = await ActiveTaskRegistry.unregister(task.task_id, run_stale)
    assert removed is False


@pytest.mark.asyncio
async def test_taskreg_8_binding_remains_after_wrong_run_unregister_attempt() -> None:
    task = _task()
    run_current = mint_run_id()
    run_stale = mint_run_id()
    await ActiveTaskRegistry.register(task, run_current)
    await ActiveTaskRegistry.unregister(task.task_id, run_stale)
    binding = await ActiveTaskRegistry.get(task.task_id)
    assert binding is not None
    assert binding.run_id == run_current
    assert binding.task is task
    assert ActiveTaskRegistry.peek_task_id_for_run(run_current) == task.task_id
    assert ActiveTaskRegistry.peek_task_id_for_run(run_stale) is None


@pytest.mark.asyncio
async def test_taskreg_8b_clear_for_tests_removes_both_indexes() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    ActiveTaskRegistry.clear_for_tests()
    assert await ActiveTaskRegistry.get(task.task_id) is None
    assert ActiveTaskRegistry.peek_task_id_for_run(run_id) is None


@pytest.mark.asyncio
async def test_taskreg_9_missing_unregister_is_harmless() -> None:
    task = _task()
    run_id = mint_run_id()
    removed = await ActiveTaskRegistry.unregister(task.task_id, run_id)
    assert removed is False


@pytest.mark.asyncio
async def test_taskreg_10_unified_task_runner_registers_canonical_run_identity() -> (
    None
):
    task = _task()
    run_id = mint_run_id()
    seen_run_id: str | None = None

    async def _handle(task: Task, *, run_id, attempt_id=None):
        nonlocal seen_run_id
        seen_run_id = run_id
        return TaskResult(
            task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED
        )

    loop = MagicMock()
    loop.handle_task = _handle
    runner = UnifiedTaskRunner(loop)  # type: ignore[arg-type]

    await runner.run_task(task, run_id=run_id)

    binding = await ActiveTaskRegistry.get(task.task_id)
    assert binding is None
    assert seen_run_id == run_id


@pytest.mark.asyncio
async def test_taskreg_11_unified_task_runner_cleanup_unregisters_same_run_identity() -> (
    None
):
    task = _task()
    run_id = mint_run_id()
    registered_run_ids: list[str] = []

    async def _handle(task: Task, *, run_id, attempt_id=None):
        binding = await ActiveTaskRegistry.get(task.task_id)
        assert binding is not None
        registered_run_ids.append(binding.run_id)
        return TaskResult(
            task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED
        )

    loop = MagicMock()
    loop.handle_task = _handle
    runner = UnifiedTaskRunner(loop)  # type: ignore[arg-type]

    await runner.run_task(task, run_id=run_id)

    assert registered_run_ids == [run_id]
    assert await ActiveTaskRegistry.get(task.task_id) is None


@pytest.mark.asyncio
async def test_taskreg_12_old_runner_cleanup_cannot_unregister_newer_binding() -> None:
    task = _task()
    run_new = mint_run_id()
    run_old = mint_run_id()
    await ActiveTaskRegistry.register(task, run_new)
    removed = await ActiveTaskRegistry.unregister(task.task_id, run_old)
    assert removed is False
    binding = await ActiveTaskRegistry.get(task.task_id)
    assert binding is not None
    assert binding.run_id == run_new


@pytest.mark.asyncio
async def test_taskreg_13_governed_cancel_active_task_targets_binding_task() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary = ControlPlaneMutationAuthorizationBoundary(
        evaluator=_AllowCancelEvaluator(),
    )
    result = await governed_cancel_active_task(
        task_id=str(task.task_id),
        run_id=str(run_id),
        mutation_id="mut-taskreg-13",
        principal=RequestIdentity(
            tenant_id=task.tenant_id,
            user_id="operator",
            principal_type=PrincipalType.USER,
            auth_subject="operator",
        ),
        mutation_boundary=boundary,
    )
    assert result.accepted is True
    assert CancellationCoordinator.is_requested(task.metadata)


class _AllowCancelEvaluator:
    def evaluate(self, request):  # type: ignore[no-untyped-def]
        del request
        return PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="test_allow",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="task_control.test_allow",
            decision_id="dec-allow",
        )


@pytest.mark.asyncio
async def test_taskreg_14_governed_set_task_autonomy_targets_binding_task() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary = ControlPlaneMutationAuthorizationBoundary(
        evaluator=_AllowCancelEvaluator()
    )
    result = await governed_set_task_autonomy(
        task_id=str(task.task_id),
        run_id=str(run_id),
        mutation_id="mut-autonomy-reg",
        target_autonomy_level=AutonomyLevel.MANUAL,
        principal=RequestIdentity(
            tenant_id=task.tenant_id,
            user_id="operator",
            principal_type=PrincipalType.USER,
            auth_subject="operator",
        ),
        mutation_boundary=boundary,
    )
    assert result.accepted is True
    assert task.options.governance.autonomy_level is AutonomyLevel.MANUAL


@pytest.mark.asyncio
async def test_taskreg_15_resolver_returns_exact_task_for_active_run() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    resolver = ActiveTaskRegistryTaskScopeResolver()
    resolved = resolver.resolve_current_task_scope(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    assert resolved == task.task_id


def test_taskreg_16_resolver_unknown_run_fails_closed() -> None:
    resolver = ActiveTaskRegistryTaskScopeResolver()
    with pytest.raises(ActiveExecutionTaskScopeUnavailable):
        resolver.resolve_current_task_scope(
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        )

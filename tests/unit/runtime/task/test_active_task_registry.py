# © Artur Czarnecki. All rights reserved.

"""TASK-CPM-0 — active execution binding for ActiveTaskRegistry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.task_control import cancel_active_task, set_task_autonomy
from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.task.active_task_registry import (
    ActiveTaskBinding,
    ActiveTaskOwnershipConflict,
    ActiveTaskRegistry,
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


@pytest.mark.asyncio
async def test_taskreg_6_unregister_exact_task_id_run_id_removes_binding() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    removed = await ActiveTaskRegistry.unregister(task.task_id, run_id)
    assert removed is True
    assert await ActiveTaskRegistry.get(task.task_id) is None


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


@pytest.mark.asyncio
async def test_taskreg_9_missing_unregister_is_harmless() -> None:
    task = _task()
    run_id = mint_run_id()
    removed = await ActiveTaskRegistry.unregister(task.task_id, run_id)
    assert removed is False


@pytest.mark.asyncio
async def test_taskreg_10_unified_task_runner_registers_canonical_run_identity() -> None:
    task = _task()
    run_id = mint_run_id()
    seen_run_id: str | None = None

    async def _handle(task: Task, *, run_id, attempt_id=None):
        nonlocal seen_run_id
        seen_run_id = run_id
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    loop = MagicMock()
    loop.handle_task = _handle
    runner = UnifiedTaskRunner(loop)  # type: ignore[arg-type]

    await runner.run_task(task, run_id=run_id)

    binding = await ActiveTaskRegistry.get(task.task_id)
    assert binding is None
    assert seen_run_id == run_id


@pytest.mark.asyncio
async def test_taskreg_11_unified_task_runner_cleanup_unregisters_same_run_identity() -> None:
    task = _task()
    run_id = mint_run_id()
    registered_run_ids: list[str] = []

    async def _handle(task: Task, *, run_id, attempt_id=None):
        binding = await ActiveTaskRegistry.get(task.task_id)
        assert binding is not None
        registered_run_ids.append(binding.run_id)
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

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
async def test_taskreg_13_cancel_active_task_targets_binding_task() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    result = await cancel_active_task(str(task.task_id))
    assert result.accepted is True
    assert CancellationCoordinator.is_requested(task.metadata)


@pytest.mark.asyncio
async def test_taskreg_14_set_task_autonomy_targets_binding_task() -> None:
    task = _task()
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    result = await set_task_autonomy(str(task.task_id), AutonomyLevel.MANUAL)
    assert result.accepted is True
    assert task.options.governance.autonomy_level is AutonomyLevel.MANUAL

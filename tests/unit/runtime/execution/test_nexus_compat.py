# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    peek_active_execution_id,
    peek_active_execution_identity,
    require_active_execution_id,
    require_active_execution_identity,
    reset_active_execution_identity,
    validate_execution_id,
)
from intergrax.runtime.execution.nexus_compat import NexusTaskExecutionDelegate
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_nexus_delegate_invokes_handle_task_once_with_resolved_identity() -> None:
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="hello",
        context=TaskContext(),
    )
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    expected = TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)
    nexus_loop = MagicMock()
    nexus_loop.handle_task = AsyncMock(return_value=expected)

    delegate = NexusTaskExecutionDelegate(
        nexus_loop,  # type: ignore[arg-type]
        run_id=run_id,
        attempt_id=attempt_id,
    )
    result = await delegate.execute(task)

    nexus_loop.handle_task.assert_awaited_once_with(
        task,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    assert result is expected


@pytest.mark.asyncio
async def test_nexus_delegate_passes_explicit_attempt_id() -> None:
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="resume",
        context=TaskContext(),
    )
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    nexus_loop = MagicMock()
    nexus_loop.handle_task = AsyncMock(
        return_value=TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)
    )

    delegate = NexusTaskExecutionDelegate(
        nexus_loop,  # type: ignore[arg-type]
        run_id=run_id,
        attempt_id=attempt_id,
    )
    await delegate.execute(task)

    nexus_loop.handle_task.assert_awaited_once_with(
        task,
        run_id=run_id,
        attempt_id=attempt_id,
    )


@pytest.mark.asyncio
async def test_nexus_preserves_boundary_execution_id(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = AgentRegistry()
    loop = NexusLoop(registry)
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    captured: dict[str, RunId | AttemptId | ExecutionId] = {}

    async def _fake_impl(task: Task) -> TaskResult:
        active_run_id, active_attempt_id = require_active_execution_identity()
        captured["run_id"] = active_run_id
        captured["attempt_id"] = active_attempt_id
        captured["execution_id"] = require_active_execution_id()
        return TaskResult(task_id=task.task_id, run_id=active_run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "_handle_task_impl", _fake_impl)
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="execute")
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    try:
        await loop.handle_task(task, run_id=run_id, attempt_id=attempt_id)
    finally:
        reset_active_execution_identity(token)

    assert captured["run_id"] == run_id
    assert captured["attempt_id"] == attempt_id
    assert captured["execution_id"] == execution_id
    assert validate_execution_id(captured["execution_id"])


@pytest.mark.asyncio
async def test_nexus_does_not_rebind_when_boundary_execution_id_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from intergrax.runtime.nexus import nexus_loop as nexus_loop_module

    bind_calls: list[tuple[RunId, AttemptId]] = []

    original_bind = bind_active_execution_identity

    def _spy_bind(*, run_id: RunId, attempt_id: AttemptId, execution_id=None):
        bind_calls.append((run_id, attempt_id))
        return original_bind(run_id=run_id, attempt_id=attempt_id, execution_id=execution_id)

    monkeypatch.setattr(nexus_loop_module, "bind_active_execution_identity", _spy_bind)

    registry = AgentRegistry()
    loop = NexusLoop(registry)
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()

    async def _fake_impl(task: Task) -> TaskResult:
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "_handle_task_impl", _fake_impl)
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="execute")
    token = nexus_loop_module.bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    try:
        await loop.handle_task(task, run_id=run_id, attempt_id=attempt_id)
    finally:
        nexus_loop_module.reset_active_execution_identity(token)

    assert bind_calls == [(run_id, attempt_id)]


@pytest.mark.asyncio
async def test_direct_legacy_nexus_call_binds_root_execution_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = AgentRegistry()
    loop = NexusLoop(registry)
    run_id = mint_run_id()
    captured: dict[str, RunId | AttemptId | ExecutionId | None] = {}

    async def _fake_impl(task: Task) -> TaskResult:
        active_run_id, active_attempt_id = require_active_execution_identity()
        captured["run_id"] = active_run_id
        captured["attempt_id"] = active_attempt_id
        captured["execution_id"] = require_active_execution_id()
        return TaskResult(task_id=task.task_id, run_id=active_run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "_handle_task_impl", _fake_impl)
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="legacy")
    await loop.handle_task(task, run_id=run_id)

    assert captured["run_id"] == run_id
    assert captured["attempt_id"] is not None
    assert validate_execution_id(captured["execution_id"])
    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None


@pytest.mark.asyncio
async def test_nexus_fails_closed_on_active_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = AgentRegistry()
    loop = NexusLoop(registry)
    run_id = mint_run_id()
    other_run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()

    async def _fake_impl(task: Task) -> TaskResult:
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "_handle_task_impl", _fake_impl)
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="mismatch")
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    try:
        with pytest.raises(RuntimeError, match="run_id mismatch"):
            await loop.handle_task(task, run_id=other_run_id, attempt_id=attempt_id)
    finally:
        reset_active_execution_identity(token)


@pytest.mark.asyncio
async def test_unified_task_runner_mints_root_execution_id_per_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, ExecutionId | AttemptId | RunId] = {}
    registry = AgentRegistry()
    loop = NexusLoop(registry)

    async def _fake_impl(task: Task) -> TaskResult:
        run_id, attempt_id = require_active_execution_identity()
        captured["run_id"] = run_id
        captured["attempt_id"] = attempt_id
        captured["execution_id"] = require_active_execution_id()
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "_handle_task_impl", _fake_impl)
    runner = UnifiedTaskRunner(loop)
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="root")
    await runner.run_task(task)

    assert validate_execution_id(captured["execution_id"])
    assert captured["attempt_id"] is not None


@pytest.mark.asyncio
async def test_unified_task_runner_passes_same_concrete_attempt_to_nexus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_attempt_ids: list[AttemptId | None] = []
    registry = AgentRegistry()
    loop = NexusLoop(registry)

    original_handle_task = loop.handle_task

    async def _spy_handle_task(
        task: Task,
        *,
        run_id: RunId,
        attempt_id: AttemptId | None = None,
    ) -> TaskResult:
        seen_attempt_ids.append(attempt_id)
        return await original_handle_task(task, run_id=run_id, attempt_id=attempt_id)

    monkeypatch.setattr(loop, "handle_task", _spy_handle_task)

    async def _fake_impl(task: Task) -> TaskResult:
        run_id, attempt_id = require_active_execution_identity()
        seen_attempt_ids.append(attempt_id)
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "_handle_task_impl", _fake_impl)
    runner = UnifiedTaskRunner(loop)
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="attempt")
    await runner.run_task(task)

    assert len(seen_attempt_ids) == 2
    assert seen_attempt_ids[0] is not None
    assert seen_attempt_ids[0] == seen_attempt_ids[1]


@pytest.mark.asyncio
async def test_resume_checkpoint_preserves_attempt_mints_fresh_execution_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from intergrax.runtime.long_running.execution_tree_checkpoint import minimal_runtime_checkpoint
    from intergrax.runtime.long_running.models import TaskCheckpoint

    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    seen_execution_ids: list[ExecutionId] = []
    registry = AgentRegistry()
    loop = NexusLoop(registry)

    async def _fake_impl(task: Task) -> TaskResult:
        active_run_id, active_attempt_id = require_active_execution_identity()
        seen_execution_ids.append(require_active_execution_id())
        assert active_run_id == run_id
        # TRANSITIONAL (UE-4B): checkpoint resume keeps AttemptId; ExecutionId is per invocation.
        assert active_attempt_id == attempt_id
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "_handle_task_impl", _fake_impl)
    runner = UnifiedTaskRunner(loop)
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="resume")
    checkpoint = TaskCheckpoint(
        task_id=task.task_id,
        tenant_id="t1",
        resume_token="rt_test",
        task_state=TaskState.WAITING_FOR_HUMAN,
        runtime=minimal_runtime_checkpoint(
            task_id=task.task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            root_execution_id=mint_execution_id(),
        ),
    )

    await runner.run_task(task, resume_checkpoint=checkpoint)
    await runner.run_task(task, resume_checkpoint=checkpoint)

    assert len(seen_execution_ids) == 2
    assert seen_execution_ids[0] != seen_execution_ids[1]
    assert validate_execution_id(seen_execution_ids[0])
    assert validate_execution_id(seen_execution_ids[1])

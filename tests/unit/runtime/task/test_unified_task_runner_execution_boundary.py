# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    validate_attempt_id,
    validate_execution_id,
)
from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver
from intergrax.runtime.execution.task_adapter import TaskExecutionInput
from intergrax.runtime.long_running.execution_tree_checkpoint import minimal_runtime_checkpoint
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _task(*, message: str = "hello") -> Task:
    return Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message=message,
        context=TaskContext(capability="echo.basic", intent="run"),
    )


def _runner_with_handle() -> tuple[UnifiedTaskRunner, MagicMock, AsyncMock]:
    handle_task = AsyncMock(
        return_value=TaskResult(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
        )
    )
    loop = MagicMock()
    loop.handle_task = handle_task
    return UnifiedTaskRunner(loop), loop, handle_task  # type: ignore[arg-type]


def test_unified_task_runner_source_has_no_direct_nexus_handle_task_call() -> None:
    source = Path("intergrax/runtime/task/unified_task_runner.py").read_text(encoding="utf-8")
    assert "self._nexus_loop.handle_task(" not in source


@pytest.mark.asyncio
async def test_run_task_reaches_nexus_exactly_once_with_same_task() -> None:
    task = _task()
    runner, _loop, handle_task = _runner_with_handle()
    handle_task.return_value = TaskResult(
        task_id=task.task_id,
        run_id=mint_run_id(),
        state=TaskState.COMPLETED,
    )

    await runner.run_task(task)

    handle_task.assert_awaited_once()
    assert handle_task.await_args.args[0] is task


@pytest.mark.asyncio
async def test_run_task_preserves_explicit_run_and_attempt_id() -> None:
    task = _task()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runner, _loop, handle_task = _runner_with_handle()

    await runner.run_task(task, run_id=run_id, attempt_id=attempt_id)

    handle_task.assert_awaited_once_with(
        task,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    passed_attempt_id = handle_task.await_args.kwargs["attempt_id"]
    assert validate_attempt_id(passed_attempt_id) == attempt_id


@pytest.mark.asyncio
async def test_run_task_mints_concrete_attempt_id_when_omitted() -> None:
    task = _task()
    runner, _loop, handle_task = _runner_with_handle()

    await runner.run_task(task)

    passed_attempt_id = handle_task.await_args.kwargs["attempt_id"]
    assert passed_attempt_id is not None
    assert validate_attempt_id(passed_attempt_id) == passed_attempt_id


@pytest.mark.asyncio
async def test_run_task_resume_checkpoint_identity_reaches_nexus() -> None:
    task = _task(message="resume")
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runner, _loop, handle_task = _runner_with_handle()
    checkpoint = TaskCheckpoint(
        task_id=task.task_id,
        tenant_id=task.tenant_id,
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

    handle_task.assert_awaited_once_with(
        task,
        run_id=run_id,
        attempt_id=attempt_id,
    )


@pytest.mark.asyncio
async def test_run_task_checkpoint_run_id_conflict_unchanged() -> None:
    task = _task()
    run_id = mint_run_id()
    other_run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runner, _loop, handle_task = _runner_with_handle()
    checkpoint = TaskCheckpoint(
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        resume_token="rt_test",
        task_state=TaskState.WAITING_FOR_HUMAN,
        runtime=minimal_runtime_checkpoint(
            task_id=task.task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            root_execution_id=mint_execution_id(),
        ),
    )

    with pytest.raises(ValueError, match="explicit run_id conflicts"):
        await runner.run_task(task, run_id=other_run_id, resume_checkpoint=checkpoint)

    handle_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_task_checkpoint_redelivery_allows_new_attempt() -> None:
    task = _task()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    other_attempt_id = mint_attempt_id()
    runner, _loop, handle_task = _runner_with_handle()
    checkpoint = TaskCheckpoint(
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        resume_token="rt_test",
        task_state=TaskState.WAITING_FOR_HUMAN,
        runtime=minimal_runtime_checkpoint(
            task_id=task.task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            root_execution_id=mint_execution_id(),
        ),
    )

    await runner.run_task(
        task,
        attempt_id=other_attempt_id,
        resume_checkpoint=checkpoint,
    )

    handle_task.assert_awaited_once()
    assert handle_task.await_args.kwargs["attempt_id"] == other_attempt_id


@pytest.mark.asyncio
async def test_run_task_task_enricher_runs_before_nexus() -> None:
    task = _task(message="original")
    seen_messages: list[str] = []

    def enricher(source: Task) -> Task:
        seen_messages.append(source.message)
        return Task(
            task_id=source.task_id,
            tenant_id=source.tenant_id,
            user_id=source.user_id,
            message="enriched",
            context=source.context,
        )

    handle_task = AsyncMock(
        return_value=TaskResult(
            task_id=task.task_id,
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
        )
    )
    loop = MagicMock()
    loop.handle_task = handle_task
    runner = UnifiedTaskRunner(loop, task_enricher=enricher)  # type: ignore[arg-type]

    await runner.run_task(task)

    assert seen_messages == ["original"]
    assert handle_task.await_args.args[0].message == "enriched"


@pytest.mark.asyncio
async def test_run_task_unregisters_on_nexus_exception() -> None:
    task = _task()
    run_id = mint_run_id()
    handle_task = AsyncMock(side_effect=RuntimeError("nexus-fail"))
    loop = MagicMock()
    loop.handle_task = handle_task
    runner = UnifiedTaskRunner(loop)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="nexus-fail"):
        await runner.run_task(task, run_id=run_id)

    assert await ActiveTaskRegistry.get(task.task_id) is None


@pytest.mark.asyncio
async def test_run_runtime_request_delegates_to_run_task() -> None:
    task = _task()
    run_id = mint_run_id()
    request = RuntimeRequest(
        agent_id="agent-1",
        user_id=task.user_id,
        session_id="sess_0123456789abcdef0123456789abcdef",
        message=task.message,
        task_id=task.task_id,
        run_id=run_id,
    )
    runner, _loop, handle_task = _runner_with_handle()
    original_run_task = runner.run_task
    calls: list[Task] = []

    async def _spy_run_task(
        incoming: Task,
        *,
        run_id=None,
        attempt_id=None,
        resume_checkpoint=None,
    ) -> TaskResult:
        calls.append(incoming)
        return await original_run_task(
            incoming,
            run_id=run_id,
            attempt_id=attempt_id,
            resume_checkpoint=resume_checkpoint,
        )

    runner.run_task = _spy_run_task  # type: ignore[method-assign]

    await runner.run_runtime_request(request, tenant_id=task.tenant_id, user_id=task.user_id)

    assert len(calls) == 1
    handle_task.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_task_strategy_resolver_receives_orchestration_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _task()
    runner, _loop, handle_task = _runner_with_handle()
    captured: list[ExecutionRequest[TaskExecutionInput, object]] = []
    original_resolve = StrategyResolver.resolve

    def _capture_resolve(
        self: StrategyResolver,
        request: ExecutionRequest[TaskExecutionInput, object],
    ) -> ExecutionStrategy:
        captured.append(request)
        return original_resolve(self, request)

    monkeypatch.setattr(StrategyResolver, "resolve", _capture_resolve)

    await runner.run_task(task)

    assert len(captured) == 1
    request = captured[0]
    assert request.capabilities == frozenset({ExecutionCapability.ORCHESTRATION})
    assert ExecutionCapability.TOOLS not in request.capabilities
    assert ExecutionCapability.STREAMING not in request.capabilities
    assert request.input.message == task.message
    assert request.input.capability == task.context.capability
    assert request.input.intent == task.context.intent
    assert StrategyResolver().resolve(request) is ExecutionStrategy.ORCHESTRATION
    handle_task.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_task_fails_closed_when_strategy_is_not_orchestration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _task()
    runner, _loop, handle_task = _runner_with_handle()

    def _wrong_strategy(
        self: StrategyResolver,
        request: ExecutionRequest[TaskExecutionInput, object],
    ) -> ExecutionStrategy:
        return ExecutionStrategy.INFERENCE

    monkeypatch.setattr(StrategyResolver, "resolve", _wrong_strategy)

    with pytest.raises(
        RuntimeError,
        match="legacy UnifiedTaskRunner compatibility path requires orchestration",
    ):
        await runner.run_task(task)

    handle_task.assert_not_awaited()
    assert await ActiveTaskRegistry.get(task.task_id) is None


@pytest.mark.asyncio
async def test_concurrent_run_task_calls_use_isolated_delegate_identity() -> None:
    task_a = _task(message="a")
    task_b = _task(message="b")
    run_id_a = mint_run_id()
    run_id_b = mint_run_id()
    attempt_id_b = mint_attempt_id()
    seen: list[tuple[str, str, str | None]] = []
    gate = asyncio.Event()

    async def _handle(task: Task, *, run_id, attempt_id=None):
        seen.append((task.message, run_id, attempt_id))
        await gate.wait()
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    loop = MagicMock()
    loop.handle_task = _handle
    runner = UnifiedTaskRunner(loop)  # type: ignore[arg-type]

    first = asyncio.create_task(runner.run_task(task_a, run_id=run_id_a))
    second = asyncio.create_task(
        runner.run_task(task_b, run_id=run_id_b, attempt_id=attempt_id_b)
    )
    await asyncio.sleep(0)
    assert ("a", run_id_a, None) not in seen
    assert any(entry[0] == "a" and entry[1] == run_id_a and entry[2] is not None for entry in seen)
    assert ("b", run_id_b, attempt_id_b) in seen
    gate.set()
    await asyncio.gather(first, second)


def test_unified_task_runner_constructor_remains_compatible() -> None:
    loop = MagicMock()
    runner = UnifiedTaskRunner(loop, task_enricher=lambda task: task)  # type: ignore[arg-type]
    assert runner.nexus_loop is loop

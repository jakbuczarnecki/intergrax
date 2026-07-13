# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from intergrax.runtime.nexus.nexus_loop import NexusLoop  # noqa: F401 — preload to avoid interactions import cycle
from intergrax.runtime.interactions.intake_service import InteractionIntakeService
from intergrax.runtime.interactions.task_executor import NexusLoopTaskExecutor
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState


class _RecordingExecutor:
    def __init__(self) -> None:
        self.prepare_calls = 0
        self.execute_calls = 0
        self.last_task: Task | None = None

    def prepare(self, task: Task) -> Task:
        self.prepare_calls += 1
        prepared = task.model_copy(
            update={"context": task.context.model_copy(update={"capability": "prepared.cap"})}
        )
        self.last_task = prepared
        return prepared

    async def execute_prepared(self, task: Task) -> TaskResult:
        self.execute_calls += 1
        self.last_task = task
        return TaskResult(task_id=task.task_id, state=TaskState.COMPLETED, answer="ok")

    async def execute(self, task: Task) -> TaskResult:
        return await self.execute_prepared(self.prepare(task))


@pytest.mark.asyncio
async def test_interaction_intake_uses_task_executor_when_execute_true() -> None:
    executor = _RecordingExecutor()
    service = InteractionIntakeService(task_executor=executor)
    intake = await service.intake_payload(
        {"message": "hello", "capability": "echo.basic", "user_id": "u1"},
        tenant_id="t1",
        execute=True,
    )
    assert intake.executed is True
    assert intake.result is not None
    assert intake.result.answer == "ok"
    assert executor.prepare_calls == 1
    assert executor.execute_calls == 1
    assert intake.task.context.capability == "prepared.cap"


@pytest.mark.asyncio
async def test_interaction_intake_does_not_execute_when_execute_false() -> None:
    executor = _RecordingExecutor()
    service = InteractionIntakeService(task_executor=executor)
    intake = await service.intake_payload(
        {"message": "hello", "capability": "echo.basic", "user_id": "u1"},
        tenant_id="t1",
        execute=False,
    )
    assert intake.executed is False
    assert intake.result is None
    assert executor.execute_calls == 0
    assert executor.prepare_calls == 1


@pytest.mark.asyncio
async def test_interaction_intake_nexus_loop_backward_compat() -> None:
    loop = AsyncMock()
    loop.handle_task.return_value = TaskResult(
        task_id="task-1",
        state=TaskState.COMPLETED,
        answer="legacy",
    )
    enricher_calls = 0

    def enricher(task: Task) -> Task:
        nonlocal enricher_calls
        enricher_calls += 1
        return task

    service = InteractionIntakeService(nexus_loop=loop, task_enricher=enricher)
    intake = await service.intake_payload(
        {"message": "hello", "capability": "echo.basic", "user_id": "u1"},
        tenant_id="t1",
        execute=True,
    )
    assert isinstance(service._resolve_executor(), NexusLoopTaskExecutor)
    assert enricher_calls == 1
    loop.handle_task.assert_awaited_once()
    assert intake.result is not None
    assert intake.result.answer == "legacy"


@pytest.mark.asyncio
async def test_interaction_intake_requires_executor_for_execute_true() -> None:
    service = InteractionIntakeService()
    with pytest.raises(ValueError, match="Task executor is not configured"):
        await service.intake_payload(
            {"message": "hello", "user_id": "u1"},
            tenant_id="t1",
            execute=True,
        )

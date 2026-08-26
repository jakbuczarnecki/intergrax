# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.execution.nexus_compat import NexusTaskExecutionDelegate
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState

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
    expected = TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)
    nexus_loop = MagicMock()
    nexus_loop.handle_task = AsyncMock(return_value=expected)

    delegate = NexusTaskExecutionDelegate(
        nexus_loop,  # type: ignore[arg-type]
        run_id=run_id,
        attempt_id=None,
    )
    result = await delegate.execute(task)

    nexus_loop.handle_task.assert_awaited_once_with(
        task,
        run_id=run_id,
        attempt_id=None,
    )
    assert result is expected


@pytest.mark.asyncio
async def test_nexus_delegate_passes_explicit_attempt_id() -> None:
    from intergrax.contracts.execution_identity import mint_attempt_id

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

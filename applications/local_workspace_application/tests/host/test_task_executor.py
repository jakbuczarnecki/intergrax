# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from intergrax.runtime.interactions.errors import HostNotAcceptingWorkError
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from local_workspace_application.host.lifecycle import LocalWorkspaceHostLifecycle
from local_workspace_application.host.lkw_task_enricher import build_lkw_application_task_enricher
from local_workspace_application.host.task_executor import LocalWorkspaceTaskExecutor
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST

pytestmark = pytest.mark.unit


@pytest.fixture
def lifecycle() -> LocalWorkspaceHostLifecycle:
    host = LocalWorkspaceHostLifecycle()
    host.set_executor_available(True)
    host.transition_to_ready()
    return host


@pytest.mark.asyncio
async def test_executor_applies_application_enrichment(lifecycle: LocalWorkspaceHostLifecycle) -> None:
    nexus = AsyncMock()
    nexus.handle_task.return_value = TaskResult(
        task_id="task-1",
        state=TaskState.COMPLETED,
        answer="ok",
    )
    env = LOCAL_WORKSPACE_APPLICATION_MANIFEST.environment
    assert env is not None
    enricher = build_lkw_application_task_enricher(
        env,
        default_capability="local.workspace.search",
    )
    executor = LocalWorkspaceTaskExecutor(nexus, task_enricher=enricher, lifecycle=lifecycle)
    task = Task(
        task_id="task-1",
        tenant_id="t1",
        user_id="u1",
        message="hello",
        context=TaskContext(),
    )
    await executor.execute(task)
    sent = nexus.handle_task.await_args.args[0]
    assert sent.context.capability == "local.workspace.search"


@pytest.mark.asyncio
async def test_executor_rejects_unsupported_capability(lifecycle: LocalWorkspaceHostLifecycle) -> None:
    nexus = AsyncMock()
    env = LOCAL_WORKSPACE_APPLICATION_MANIFEST.environment
    assert env is not None
    enricher = build_lkw_application_task_enricher(
        env,
        default_capability="local.workspace.search",
    )
    executor = LocalWorkspaceTaskExecutor(nexus, task_enricher=enricher, lifecycle=lifecycle)
    task = Task(
        task_id="task-1",
        tenant_id="t1",
        user_id="u1",
        message="hello",
        context=TaskContext(capability="echo.basic"),
    )
    with pytest.raises(ValueError, match="unsupported_lkw_capability"):
        await executor.execute(task)
    nexus.handle_task.assert_not_called()


@pytest.mark.asyncio
async def test_executor_rejects_when_host_not_ready() -> None:
    lifecycle = LocalWorkspaceHostLifecycle()
    lifecycle.set_executor_available(True)
    lifecycle.transition_to_ready()
    lifecycle.transition_to_stopping()
    nexus = AsyncMock()
    executor = LocalWorkspaceTaskExecutor(nexus, task_enricher=None, lifecycle=lifecycle)
    task = Task(task_id="t1", tenant_id="t1", user_id="u1", message="m")
    with pytest.raises(HostNotAcceptingWorkError) as exc:
        await executor.execute(task)
    assert exc.value.error_id == "lkw_host_stopping"


@pytest.mark.asyncio
async def test_executor_delegates_to_nexus_once(lifecycle: LocalWorkspaceHostLifecycle) -> None:
    nexus = AsyncMock()
    nexus.handle_task.return_value = TaskResult(
        task_id="task-1",
        state=TaskState.COMPLETED,
        answer="ok",
    )
    executor = LocalWorkspaceTaskExecutor(nexus, task_enricher=None, lifecycle=lifecycle)
    task = Task(
        task_id="task-1",
        tenant_id="t1",
        user_id="u1",
        message="hello",
        context=TaskContext(capability="local.workspace.search"),
    )
    result = await executor.execute(task)
    nexus.handle_task.assert_awaited_once()
    assert result.answer == "ok"

# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock

import pytest

from intergrax.contracts.execution_identity import mint_task_id
from intergrax.runtime.interactions.errors import HostNotAcceptingWorkError
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from local_workspace_application.host.lifecycle import LocalWorkspaceHostLifecycle
from local_workspace_application.host.lkw_task_enricher import build_lkw_application_task_enricher
from local_workspace_application.host.readiness import (
    LocalWorkspaceComponentReadiness,
    LocalWorkspaceReadinessSnapshot,
)
from local_workspace_application.host.task_executor import LocalWorkspaceTaskExecutor
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST

pytestmark = pytest.mark.unit


@dataclass
class _FakeReadiness:
    snapshot: LocalWorkspaceReadinessSnapshot
    calls: list[LocalWorkspaceReadinessSnapshot] = field(default_factory=list)

    def readiness_snapshot(self) -> LocalWorkspaceReadinessSnapshot:
        self.calls.append(self.snapshot)
        return self.snapshot


def _nexus_loop_mock() -> AsyncMock:
    return AsyncMock()


def _host_execution_mock(*, execute_result: TaskResult) -> AsyncMock:
    host_execution = AsyncMock()
    host_execution.execute = AsyncMock(return_value=execute_result)
    return host_execution


@pytest.fixture
def lifecycle() -> LocalWorkspaceHostLifecycle:
    host = LocalWorkspaceHostLifecycle()
    host.set_executor_available(True)
    host.transition_to_ready()
    return host


@pytest.mark.asyncio
async def test_executor_applies_application_enrichment(lifecycle: LocalWorkspaceHostLifecycle) -> None:
    host_execution = _host_execution_mock(
        execute_result=TaskResult(
            task_id="task-1",
            state=TaskState.COMPLETED,
            answer="ok",
        ),
    )
    env = LOCAL_WORKSPACE_APPLICATION_MANIFEST.environment
    assert env is not None
    enricher = build_lkw_application_task_enricher(
        env,
        default_capability="local.workspace.search",
    )
    executor = LocalWorkspaceTaskExecutor(
        host_execution,
        nexus_loop=_nexus_loop_mock(),
        task_enricher=enricher,
        readiness=lifecycle,
    )
    task = Task(
        task_id=mint_task_id(),
        tenant_id="t1",
        user_id="u1",
        message="hello",
        context=TaskContext(),
    )
    await executor.execute(task)
    sent = host_execution.execute.await_args.args[0]
    assert sent.context.capability == "local.workspace.search"


@pytest.mark.asyncio
async def test_executor_rejects_unsupported_capability(lifecycle: LocalWorkspaceHostLifecycle) -> None:
    host_execution = _host_execution_mock(
        execute_result=TaskResult(task_id="task-1", state=TaskState.COMPLETED),
    )
    env = LOCAL_WORKSPACE_APPLICATION_MANIFEST.environment
    assert env is not None
    enricher = build_lkw_application_task_enricher(
        env,
        default_capability="local.workspace.search",
    )
    executor = LocalWorkspaceTaskExecutor(
        host_execution,
        nexus_loop=_nexus_loop_mock(),
        task_enricher=enricher,
        readiness=lifecycle,
    )
    task = Task(
        task_id=mint_task_id(),
        tenant_id="t1",
        user_id="u1",
        message="hello",
        context=TaskContext(capability="echo.basic"),
    )
    with pytest.raises(ValueError, match="unsupported_lkw_capability"):
        await executor.execute(task)
    host_execution.execute.assert_not_called()


@pytest.mark.asyncio
async def test_executor_rejects_when_host_not_ready() -> None:
    lifecycle = LocalWorkspaceHostLifecycle()
    lifecycle.set_executor_available(True)
    lifecycle.transition_to_ready()
    lifecycle.transition_to_stopping()
    host_execution = _host_execution_mock(
        execute_result=TaskResult(task_id="t1", state=TaskState.COMPLETED),
    )
    executor = LocalWorkspaceTaskExecutor(
        host_execution,
        nexus_loop=_nexus_loop_mock(),
        task_enricher=None,
        readiness=lifecycle,
    )
    task = Task(task_id=mint_task_id(), tenant_id="t1", user_id="u1", message="m")
    with pytest.raises(HostNotAcceptingWorkError) as exc:
        await executor.execute(task)
    assert exc.value.error_id == "lkw_host_stopping"


@pytest.mark.asyncio
async def test_executor_delegates_to_host_execution_once(
    lifecycle: LocalWorkspaceHostLifecycle,
) -> None:
    host_execution = _host_execution_mock(
        execute_result=TaskResult(
            task_id="task-1",
            state=TaskState.COMPLETED,
            answer="ok",
        ),
    )
    executor = LocalWorkspaceTaskExecutor(
        host_execution,
        nexus_loop=_nexus_loop_mock(),
        task_enricher=None,
        readiness=lifecycle,
    )
    task = Task(
        task_id=mint_task_id(),
        tenant_id="t1",
        user_id="u1",
        message="hello",
        context=TaskContext(capability="local.workspace.search"),
    )
    result = await executor.execute(task)
    host_execution.execute.assert_awaited_once()
    assert result.answer == "ok"


@pytest.mark.asyncio
async def test_executor_uses_readiness_provider_snapshot() -> None:
    accepting = LocalWorkspaceReadinessSnapshot(
        ready=True,
        accepts_new_work=True,
        state="ready",
        detail="ready",
        rejection_error_id="",
        components=(
            LocalWorkspaceComponentReadiness(
                name="runtime",
                enabled=True,
                required=True,
                healthy=True,
            ),
        ),
    )
    readiness = _FakeReadiness(snapshot=accepting)
    host_execution = _host_execution_mock(
        execute_result=TaskResult(
            task_id="task-1",
            state=TaskState.COMPLETED,
            answer="ok",
        ),
    )
    executor = LocalWorkspaceTaskExecutor(
        host_execution,
        nexus_loop=_nexus_loop_mock(),
        task_enricher=None,
        readiness=readiness,
    )
    task = Task(
        task_id=mint_task_id(),
        tenant_id="t1",
        user_id="u1",
        message="hello",
        context=TaskContext(capability="local.workspace.search"),
    )
    await executor.execute(task)
    host_execution.execute.assert_awaited_once()
    assert len(readiness.calls) == 1

    readiness.snapshot = LocalWorkspaceReadinessSnapshot(
        ready=False,
        accepts_new_work=False,
        state="starting",
        detail="host_state=starting",
        rejection_error_id="lkw_host_not_ready",
    )
    with pytest.raises(HostNotAcceptingWorkError) as not_ready:
        await executor.execute(task)
    assert not_ready.value.error_id == "lkw_host_not_ready"
    assert not_ready.value.detail == "host_state=starting"

    readiness.snapshot = LocalWorkspaceReadinessSnapshot(
        ready=False,
        accepts_new_work=False,
        state="stopping",
        detail="host_state=stopping",
        rejection_error_id="lkw_host_stopping",
    )
    with pytest.raises(HostNotAcceptingWorkError) as stopping:
        await executor.execute(task)
    assert stopping.value.error_id == "lkw_host_stopping"
    assert stopping.value.detail == "host_state=stopping"

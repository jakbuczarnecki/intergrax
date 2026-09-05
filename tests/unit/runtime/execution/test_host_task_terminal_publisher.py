# © Artur Czarnecki. All rights reserved.

"""Host task terminal publisher port and HostTaskExecution boundary tests (NPSC-3B-R1)."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, fields
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from intergrax.applications._shared.host_task_execution_wiring import (
    build_host_task_execution,
    build_nexus_host_task_terminal_publisher,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
    validate_execution_id,
)
from intergrax.runtime.execution.host_task import HostTaskExecution
from intergrax.runtime.execution.host_task_terminal_publisher import HostTaskTerminalPublisher
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_HOST_TASK_PATH = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "host_task.py"


@dataclass(frozen=True, slots=True)
class RecordingTerminalPublisher:
    calls: list[tuple[Task, RunId, AttemptId, ExecutionId]]

    async def publish_terminal(
        self,
        task: Task,
        *,
        run_id: RunId,
        attempt_id: AttemptId,
        execution_id: ExecutionId,
    ) -> None:
        self.calls.append((task, run_id, attempt_id, execution_id))


def _build_host_task_execution(
    *,
    terminal_publisher: HostTaskTerminalPublisher | None = None,
) -> HostTaskExecution:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    execution = build_host_task_execution(nexus_loop, orchestration_triggers=frozenset())
    if terminal_publisher is None:
        return execution
    return HostTaskExecution(
        _agent_engine=execution._agent_engine,
        _agent_router=execution._agent_router,
        _orchestration_executor=execution._orchestration_executor,
        _orchestration_triggers=execution._orchestration_triggers,
        _pipeline_capability_suffix=execution._pipeline_capability_suffix,
        _ledger_factory=execution._ledger_factory,
        _run_budget=execution._run_budget,
        _terminal_publisher=terminal_publisher,
    )


def _sample_task() -> Task:
    return Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="terminal publisher proof",
        context=TaskContext(capability="echo.basic"),
        agent_id="search",
    )


def test_host_task_module_has_no_nexus_dependency() -> None:
    text = _HOST_TASK_PATH.read_text(encoding="utf-8")
    assert "NexusLoop" not in text
    assert "nexus_loop" not in text
    assert "_nexus_loop" not in text
    assert "publish_host_task_terminal_runtime" not in text


def test_host_task_execution_has_terminal_publisher_field_only() -> None:
    field_names = {field.name for field in fields(HostTaskExecution)}
    assert "_terminal_publisher" in field_names
    assert "_revision_admission" in field_names
    assert "_nexus_loop" not in field_names
    assert "nexus_loop" not in field_names


def test_host_task_execution_accepts_terminal_publisher_port() -> None:
    publisher = RecordingTerminalPublisher(calls=[])
    execution = _build_host_task_execution(terminal_publisher=publisher)
    assert execution._terminal_publisher is publisher


@pytest.mark.asyncio
async def test_completed_task_result_publishes_terminal_once() -> None:
    publisher = RecordingTerminalPublisher(calls=[])
    execution = _build_host_task_execution(terminal_publisher=publisher)
    task = _sample_task()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()

    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        new_callable=AsyncMock,
        return_value=TaskResult(
            task_id=task.task_id,
            run_id=run_id,
            state=TaskState.COMPLETED,
            answer="ok",
            agent_id=task.agent_id,
        ),
    ):
        result = await execution.execute(task, run_id=run_id, attempt_id=attempt_id)

    assert result.state is TaskState.COMPLETED
    assert len(publisher.calls) == 1
    published_task, published_run_id, published_attempt_id, published_execution_id = publisher.calls[0]
    assert published_task.task_id == task.task_id
    assert published_task.state is TaskState.COMPLETED
    assert published_task.agent_id == task.agent_id
    assert published_run_id == run_id
    assert published_attempt_id == attempt_id
    validate_execution_id(published_execution_id)


@pytest.mark.asyncio
async def test_failed_task_result_publishes_terminal_once() -> None:
    publisher = RecordingTerminalPublisher(calls=[])
    execution = _build_host_task_execution(terminal_publisher=publisher)
    task = _sample_task()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()

    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        new_callable=AsyncMock,
        return_value=TaskResult(
            task_id=task.task_id,
            run_id=run_id,
            state=TaskState.FAILED,
            answer="failed",
            agent_id=task.agent_id,
        ),
    ):
        result = await execution.execute(task, run_id=run_id, attempt_id=attempt_id)

    assert result.state is TaskState.FAILED
    assert len(publisher.calls) == 1
    published_task, _, _, _ = publisher.calls[0]
    assert published_task.state is TaskState.FAILED


@pytest.mark.asyncio
async def test_non_terminal_task_result_does_not_publish() -> None:
    publisher = RecordingTerminalPublisher(calls=[])
    execution = _build_host_task_execution(terminal_publisher=publisher)
    task = _sample_task()

    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        new_callable=AsyncMock,
        return_value=TaskResult(
            task_id=task.task_id,
            run_id=mint_run_id(),
            state=TaskState.RUNNING,
            answer="running",
            agent_id=task.agent_id,
        ),
    ):
        result = await execution.execute(task)

    assert result.state is TaskState.RUNNING
    assert publisher.calls == []


@pytest.mark.asyncio
async def test_missing_terminal_publisher_executes_without_error() -> None:
    execution = _build_host_task_execution(terminal_publisher=None)
    task = _sample_task()

    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        new_callable=AsyncMock,
        return_value=TaskResult(
            task_id=task.task_id,
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="ok",
            agent_id=task.agent_id,
        ),
    ):
        result = await execution.execute(task)

    assert result.state is TaskState.COMPLETED


@pytest.mark.asyncio
async def test_nexus_adapter_delegates_terminal_publication() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    execution = build_host_task_execution(nexus_loop, orchestration_triggers=frozenset())
    task = _sample_task()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()

    with patch.object(
        nexus_loop,
        "publish_host_task_terminal_runtime",
        new_callable=AsyncMock,
    ) as publish_mock:
        with patch(
            "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
            new_callable=AsyncMock,
            return_value=TaskResult(
                task_id=task.task_id,
                run_id=run_id,
                state=TaskState.COMPLETED,
                answer="ok",
                agent_id=task.agent_id,
            ),
        ):
            await execution.execute(task, run_id=run_id, attempt_id=attempt_id)

    publish_mock.assert_awaited_once()
    kwargs = publish_mock.await_args.kwargs
    assert kwargs["run_id"] == run_id
    assert kwargs["attempt_id"] == attempt_id
    validate_execution_id(kwargs["execution_id"])
    published_task = publish_mock.await_args.args[0]
    assert published_task.task_id == task.task_id
    assert published_task.state is TaskState.COMPLETED


def test_nexus_adapter_is_typed_port_implementation() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    publisher = build_nexus_host_task_terminal_publisher(nexus_loop)
    assert inspect.iscoroutinefunction(publisher.publish_terminal)

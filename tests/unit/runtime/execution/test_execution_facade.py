# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.execution_identity import mint_task_id
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.runtime.execution import Execution
from intergrax.runtime.execution.boundary import ExecutionBoundary
from intergrax.runtime.execution.runtime import ExecutionRuntime, RootExecutionOptions, resolve_root_execution_context
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass(frozen=True)
class Ping:
    value: str


@dataclass(frozen=True)
class Pong:
    value: str


class CountingPingDelegate:
    def __init__(self, result: Pong) -> None:
        self.call_count = 0
        self.last_request: Ping | None = None
        self._result = result

    async def execute(self, request: Ping) -> Pong:
        self.call_count += 1
        self.last_request = request
        return self._result


class FailingPingDelegate:
    async def execute(self, request: Ping) -> Pong:
        raise ValueError(f"boom:{request.value}")


class FakeTaskRunner:
    def __init__(self, result: TaskResult) -> None:
        self.call_count = 0
        self.last_task: Task | None = None
        self._result = result

    async def run_task(self, task: Task) -> TaskResult:
        self.call_count += 1
        self.last_task = task
        return self._result


class TaskExecutionDelegate:
    def __init__(self, runner: FakeTaskRunner) -> None:
        self._runner = runner

    async def execute(self, task: Task) -> TaskResult:
        return await self._runner.run_task(task)


def _minimal_task() -> Task:
    return Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="hello",
        context=TaskContext(capability="echo.basic"),
    )


def _minimal_task_result() -> TaskResult:
    return TaskResult(task_id=mint_task_id(), state=TaskState.COMPLETED, answer="ok")


def _root_context() -> object:
    return resolve_root_execution_context(
        RootExecutionOptions(authority=ParentExecutionAuthority.unrestricted_root())
    )


def _ping_execution(result: Pong | None = None) -> Execution[Ping, Pong]:
    expected = result or Pong(value="pong")
    runtime = ExecutionRuntime[Ping, Pong](CountingPingDelegate(expected))
    return Execution[Ping, Pong](runtime)


@pytest.mark.asyncio
async def test_facade_delegates_typed_request_to_boundary_exactly_once() -> None:
    delegate = CountingPingDelegate(Pong(value="pong"))
    runtime = ExecutionRuntime[Ping, Pong](delegate)
    execution = Execution[Ping, Pong](runtime)
    request = Ping(value="ping")
    context = _root_context()

    await execution.execute(request, root_context=context)

    assert delegate.call_count == 1
    assert delegate.last_request == request


@pytest.mark.asyncio
async def test_facade_returns_exact_boundary_result() -> None:
    expected = Pong(value="exact")
    execution = _ping_execution(expected)

    result = await execution.execute(Ping(value="ping"), root_context=_root_context())

    assert result is expected


@pytest.mark.asyncio
async def test_facade_propagates_boundary_exception_unchanged() -> None:
    boundary = ExecutionBoundary[Ping, Pong](FailingPingDelegate())
    runtime = ExecutionRuntime[Ping, Pong](boundary)
    execution = Execution[Ping, Pong](runtime)

    with pytest.raises(ValueError, match="boom:fail"):
        await execution.execute(Ping(value="fail"), root_context=_root_context())


@pytest.mark.asyncio
async def test_facade_does_not_retry_on_boundary_failure() -> None:
    class RetryObservingDelegate:
        def __init__(self) -> None:
            self.call_count = 0

        async def execute(self, request: Ping) -> Pong:
            self.call_count += 1
            raise RuntimeError("no-retry")

    delegate = RetryObservingDelegate()
    runtime = ExecutionRuntime[Ping, Pong](delegate)
    execution = Execution[Ping, Pong](runtime)

    with pytest.raises(RuntimeError, match="no-retry"):
        await execution.execute(Ping(value="once"), root_context=_root_context())

    assert delegate.call_count == 1


@pytest.mark.asyncio
async def test_facade_works_with_non_task_typed_dtos() -> None:
    execution = _ping_execution(Pong(value="typed-pong"))

    result = await execution.execute(Ping(value="typed-ping"), root_context=_root_context())

    assert result == Pong(value="typed-pong")


@pytest.mark.asyncio
async def test_task_typed_boundary_composition() -> None:
    task = _minimal_task()
    expected = _minimal_task_result()
    runner = FakeTaskRunner(expected)
    delegate = TaskExecutionDelegate(runner)
    runtime = ExecutionRuntime[Task, TaskResult](delegate)
    execution = Execution[Task, TaskResult](runtime)

    result = await execution.execute(task, root_context=_root_context())

    assert runner.call_count == 1
    assert runner.last_task is task
    assert result is expected


@pytest.mark.asyncio
async def test_facade_instances_do_not_share_mutable_state() -> None:
    delegate_a = CountingPingDelegate(Pong(value="a"))
    delegate_b = CountingPingDelegate(Pong(value="b"))
    execution_a = Execution[Ping, Pong](ExecutionRuntime[Ping, Pong](delegate_a))
    execution_b = Execution[Ping, Pong](ExecutionRuntime[Ping, Pong](delegate_b))
    context = _root_context()

    await execution_a.execute(Ping(value="a"), root_context=context)
    await execution_b.execute(Ping(value="b"), root_context=context)

    assert delegate_a.call_count == 1
    assert delegate_b.call_count == 1
    assert delegate_a.last_request == Ping(value="a")
    assert delegate_b.last_request == Ping(value="b")


def test_package_root_exports_execution() -> None:
    from intergrax.runtime.execution import Execution as ExportedExecution

    assert ExportedExecution is Execution

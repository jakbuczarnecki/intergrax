# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import mint_task_id
from intergrax.runtime.execution.boundary import ExecutionBoundary
from intergrax.runtime.execution.task_compat import UnifiedTaskRunnerExecutionDelegate
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORBIDDEN_BOUNDARY_IMPORT_PREFIXES = (
    "intergrax.runtime.nexus",
    "intergrax.agents",
    "intergrax.runtime.policy",
    "intergrax.runtime.observability",
    "intergrax.runtime.long_running",
    "intergrax.runtime.background_execution",
    "intergrax.runtime.nexus.tools",
)


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


class ExplicitPingDelegate:
    """Explicit concrete implementation of the delegate contract."""

    async def execute(self, request: Ping) -> Pong:
        return Pong(value=f"explicit:{request.value}")


class FakeTaskRunner:
    def __init__(self, result: TaskResult) -> None:
        self.call_count = 0
        self.last_task: Task | None = None
        self._result = result

    async def run_task(self, task: Task) -> TaskResult:
        self.call_count += 1
        self.last_task = task
        return self._result


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


@pytest.mark.asyncio
async def test_boundary_delegates_typed_request_exactly_once() -> None:
    delegate = CountingPingDelegate(Pong(value="pong"))
    boundary = ExecutionBoundary[Ping, Pong](delegate)
    request = Ping(value="ping")

    await boundary.execute(request)

    assert delegate.call_count == 1
    assert delegate.last_request == request


@pytest.mark.asyncio
async def test_boundary_returns_exact_delegate_result() -> None:
    expected = Pong(value="exact")
    boundary = ExecutionBoundary[Ping, Pong](CountingPingDelegate(expected))

    result = await boundary.execute(Ping(value="ping"))

    assert result is expected


@pytest.mark.asyncio
async def test_boundary_propagates_delegate_exception_unchanged() -> None:
    boundary = ExecutionBoundary[Ping, Pong](FailingPingDelegate())

    with pytest.raises(ValueError, match="boom:fail"):
        await boundary.execute(Ping(value="fail"))


@pytest.mark.asyncio
async def test_boundary_does_not_retry_on_delegate_failure() -> None:
    class RetryObservingDelegate:
        def __init__(self) -> None:
            self.call_count = 0

        async def execute(self, request: Ping) -> Pong:
            self.call_count += 1
            raise RuntimeError("no-retry")

    delegate = RetryObservingDelegate()
    boundary = ExecutionBoundary[Ping, Pong](delegate)

    with pytest.raises(RuntimeError, match="no-retry"):
        await boundary.execute(Ping(value="once"))

    assert delegate.call_count == 1


@pytest.mark.asyncio
async def test_boundary_instances_do_not_share_mutable_runtime_state() -> None:
    delegate_a = CountingPingDelegate(Pong(value="a"))
    delegate_b = CountingPingDelegate(Pong(value="b"))
    boundary_a = ExecutionBoundary[Ping, Pong](delegate_a)
    boundary_b = ExecutionBoundary[Ping, Pong](delegate_b)

    await boundary_a.execute(Ping(value="a"))
    await boundary_b.execute(Ping(value="b"))

    assert delegate_a.call_count == 1
    assert delegate_b.call_count == 1
    assert delegate_a.last_request == Ping(value="a")
    assert delegate_b.last_request == Ping(value="b")


@pytest.mark.asyncio
async def test_explicit_class_can_satisfy_execution_delegate_contract() -> None:
    boundary = ExecutionBoundary[Ping, Pong](ExplicitPingDelegate())

    result = await boundary.execute(Ping(value="typed"))

    assert result == Pong(value="explicit:typed")


@pytest.mark.asyncio
async def test_unified_task_runner_compat_delegate_invokes_runner_once() -> None:
    task = _minimal_task()
    expected = _minimal_task_result()
    runner = FakeTaskRunner(expected)
    delegate = UnifiedTaskRunnerExecutionDelegate(runner)
    boundary = ExecutionBoundary[Task, TaskResult](delegate)

    result = await boundary.execute(task)

    assert runner.call_count == 1
    assert runner.last_task is task
    assert result is expected


def test_core_boundary_module_has_no_forbidden_imports() -> None:
    boundary_path = Path("intergrax/runtime/execution/boundary.py")
    module = ast.parse(boundary_path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)

    for forbidden in _FORBIDDEN_BOUNDARY_IMPORT_PREFIXES:
        assert not any(
            name == forbidden or name.startswith(f"{forbidden}.") for name in imported
        ), f"forbidden import in boundary.py: {forbidden}"

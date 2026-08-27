# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    peek_active_execution_id,
    peek_active_execution_identity,
    peek_active_parent_execution_id,
    require_active_execution_id,
    require_active_execution_identity,
    validate_execution_id,
)
from intergrax.runtime.execution import __all__ as execution_public_api
from intergrax.runtime.execution.boundary import (
    ExecutionAdmissionHook,
    ExecutionBoundary,
    ExecutionIdentityBinding,
)
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


class RecordingAdmissionHook:
    def __init__(self, events: list[str], label: str) -> None:
        self._events = events
        self._label = label
        self.last_request: Ping | None = None

    async def admit(self, request: Ping) -> None:
        self._events.append(f"admit:{self._label}")
        self.last_request = request


class FailingAdmissionHook:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def admit(self, request: Ping) -> None:
        raise self._exc


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


@pytest.mark.asyncio
async def test_boundary_with_zero_hooks_invokes_delegate_once() -> None:
    delegate = CountingPingDelegate(Pong(value="pong"))
    boundary = ExecutionBoundary[Ping, Pong](delegate)

    await boundary.execute(Ping(value="ping"))

    assert delegate.call_count == 1


@pytest.mark.asyncio
async def test_boundary_runs_single_admission_hook_before_delegate() -> None:
    events: list[str] = []
    delegate = CountingPingDelegate(Pong(value="pong"))
    hook = RecordingAdmissionHook(events, "only")
    boundary = ExecutionBoundary[Ping, Pong](delegate, admission_hooks=(hook,))

    await boundary.execute(Ping(value="ping"))

    assert events == ["admit:only"]
    assert delegate.call_count == 1


@pytest.mark.asyncio
async def test_boundary_runs_multiple_admission_hooks_in_tuple_order() -> None:
    events: list[str] = []
    delegate = CountingPingDelegate(Pong(value="pong"))
    boundary = ExecutionBoundary[Ping, Pong](
        delegate,
        admission_hooks=(
            RecordingAdmissionHook(events, "first"),
            RecordingAdmissionHook(events, "second"),
            RecordingAdmissionHook(events, "third"),
        ),
    )

    await boundary.execute(Ping(value="ping"))

    assert events == ["admit:first", "admit:second", "admit:third"]
    assert delegate.call_count == 1


@pytest.mark.asyncio
async def test_boundary_passes_exact_same_request_to_hooks_and_delegate() -> None:
    request = Ping(value="shared")
    events: list[str] = []
    hook_a = RecordingAdmissionHook(events, "a")
    hook_b = RecordingAdmissionHook(events, "b")
    delegate = CountingPingDelegate(Pong(value="pong"))
    boundary = ExecutionBoundary[Ping, Pong](
        delegate,
        admission_hooks=(hook_a, hook_b),
    )

    await boundary.execute(request)

    assert hook_a.last_request is request
    assert hook_b.last_request is request
    assert delegate.last_request is request


@pytest.mark.asyncio
async def test_admission_hooks_do_not_transform_request_or_result() -> None:
    request = Ping(value="unchanged")
    expected = Pong(value="unchanged-result")
    boundary = ExecutionBoundary[Ping, Pong](
        CountingPingDelegate(expected),
        admission_hooks=(RecordingAdmissionHook([], "noop"),),
    )

    result = await boundary.execute(request)

    assert result is expected


@pytest.mark.asyncio
async def test_first_admission_hook_failure_skips_later_hooks_and_delegate() -> None:
    events: list[str] = []
    delegate = CountingPingDelegate(Pong(value="pong"))
    exc = RuntimeError("first-hook-fail")
    boundary = ExecutionBoundary[Ping, Pong](
        delegate,
        admission_hooks=(
            FailingAdmissionHook(exc),
            RecordingAdmissionHook(events, "second"),
        ),
    )

    with pytest.raises(RuntimeError) as raised:
        await boundary.execute(Ping(value="ping"))

    assert raised.value is exc
    assert events == []
    assert delegate.call_count == 0


@pytest.mark.asyncio
async def test_middle_admission_hook_failure_skips_later_hooks_and_delegate() -> None:
    events: list[str] = []
    delegate = CountingPingDelegate(Pong(value="pong"))
    exc = ValueError("middle-hook-fail")
    boundary = ExecutionBoundary[Ping, Pong](
        delegate,
        admission_hooks=(
            RecordingAdmissionHook(events, "first"),
            FailingAdmissionHook(exc),
            RecordingAdmissionHook(events, "third"),
        ),
    )

    with pytest.raises(ValueError) as raised:
        await boundary.execute(Ping(value="ping"))

    assert raised.value is exc
    assert events == ["admit:first"]
    assert delegate.call_count == 0


@pytest.mark.asyncio
async def test_boundary_propagates_delegate_exception_after_successful_admission() -> None:
    events: list[str] = []
    boundary = ExecutionBoundary[Ping, Pong](
        FailingPingDelegate(),
        admission_hooks=(RecordingAdmissionHook(events, "ok"),),
    )

    with pytest.raises(ValueError, match="boom:fail"):
        await boundary.execute(Ping(value="fail"))

    assert events == ["admit:ok"]


def test_boundary_delegate_only_constructor_remains_compatible() -> None:
    delegate = CountingPingDelegate(Pong(value="pong"))
    boundary = ExecutionBoundary[Ping, Pong](delegate)

    assert boundary._admission_hooks == ()
    assert boundary._identity is None


class IdentityProbingAdmissionHook:
    def __init__(self, captured: dict[str, RunId | AttemptId | ExecutionId]) -> None:
        self._captured = captured

    async def admit(self, request: Ping) -> None:
        run_id, attempt_id = require_active_execution_identity()
        execution_id = require_active_execution_id()
        self._captured["hook_run_id"] = run_id
        self._captured["hook_attempt_id"] = attempt_id
        self._captured["hook_execution_id"] = execution_id


class IdentityProbingDelegate:
    def __init__(self, captured: dict[str, RunId | AttemptId | ExecutionId], result: Pong) -> None:
        self._captured = captured
        self._result = result

    async def execute(self, request: Ping) -> Pong:
        run_id, attempt_id = require_active_execution_identity()
        execution_id = require_active_execution_id()
        self._captured["delegate_run_id"] = run_id
        self._captured["delegate_attempt_id"] = attempt_id
        self._captured["delegate_execution_id"] = execution_id
        return self._result


def _identity_binding() -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


@pytest.mark.asyncio
async def test_boundary_binds_parent_execution_id_for_child_identity() -> None:
    parent_execution_id = mint_execution_id()
    identity = ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        parent_execution_id=parent_execution_id,
    )
    captured: dict[str, ExecutionId | None] = {}

    class ParentProbingHook:
        async def admit(self, request: Ping) -> None:
            captured["hook_parent"] = peek_active_parent_execution_id()

    class ParentProbingDelegate:
        async def execute(self, request: Ping) -> Pong:
            captured["delegate_parent"] = peek_active_parent_execution_id()
            return Pong(value="pong")

    boundary = ExecutionBoundary[Ping, Pong](
        ParentProbingDelegate(),
        admission_hooks=(ParentProbingHook(),),
        identity=identity,
    )

    await boundary.execute(Ping(value="ping"))

    assert captured["hook_parent"] == parent_execution_id
    assert captured["delegate_parent"] == parent_execution_id


@pytest.mark.asyncio
async def test_boundary_root_identity_has_no_parent_execution_id() -> None:
    identity = _identity_binding()
    captured: dict[str, ExecutionId | None] = {}

    class RootProbeDelegate:
        async def execute(self, request: Ping) -> Pong:
            captured["parent"] = peek_active_parent_execution_id()
            return Pong(value="pong")

    await ExecutionBoundary[Ping, Pong](
        RootProbeDelegate(),
        identity=identity,
    ).execute(Ping(value="ping"))

    assert captured["parent"] is None


@pytest.mark.asyncio
async def test_boundary_binds_identity_before_first_admission_hook() -> None:
    identity = _identity_binding()
    captured: dict[str, RunId | AttemptId | ExecutionId] = {}
    events: list[str] = []

    class OrderingHook:
        async def admit(self, request: Ping) -> None:
            events.append("hook")
            run_id, attempt_id = require_active_execution_identity()
            execution_id = require_active_execution_id()
            captured["hook_run_id"] = run_id
            captured["hook_attempt_id"] = attempt_id
            captured["hook_execution_id"] = execution_id

    delegate = CountingPingDelegate(Pong(value="pong"))
    boundary = ExecutionBoundary[Ping, Pong](
        delegate,
        admission_hooks=(OrderingHook(),),
        identity=identity,
    )

    await boundary.execute(Ping(value="ping"))

    assert events == ["hook"]
    assert captured["hook_run_id"] == identity.run_id
    assert captured["hook_attempt_id"] == identity.attempt_id
    assert captured["hook_execution_id"] == identity.execution_id


@pytest.mark.asyncio
async def test_boundary_hook_and_delegate_see_exact_identity() -> None:
    identity = _identity_binding()
    captured: dict[str, RunId | AttemptId | ExecutionId] = {}
    boundary = ExecutionBoundary[Ping, Pong](
        IdentityProbingDelegate(captured, Pong(value="pong")),
        admission_hooks=(IdentityProbingAdmissionHook(captured),),
        identity=identity,
    )

    await boundary.execute(Ping(value="ping"))

    assert captured["hook_run_id"] == identity.run_id
    assert captured["hook_attempt_id"] == identity.attempt_id
    assert captured["hook_execution_id"] == identity.execution_id
    assert captured["delegate_run_id"] == identity.run_id
    assert captured["delegate_attempt_id"] == identity.attempt_id
    assert captured["delegate_execution_id"] == identity.execution_id
    assert validate_execution_id(captured["hook_execution_id"])
    assert captured["hook_execution_id"] == captured["delegate_execution_id"]


@pytest.mark.asyncio
async def test_boundary_resets_identity_after_success() -> None:
    identity = _identity_binding()
    boundary = ExecutionBoundary[Ping, Pong](
        CountingPingDelegate(Pong(value="pong")),
        identity=identity,
    )

    await boundary.execute(Ping(value="ping"))

    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None


@pytest.mark.asyncio
async def test_boundary_resets_identity_after_admission_exception() -> None:
    identity = _identity_binding()
    boundary = ExecutionBoundary[Ping, Pong](
        CountingPingDelegate(Pong(value="pong")),
        admission_hooks=(FailingAdmissionHook(RuntimeError("admission-fail")),),
        identity=identity,
    )

    with pytest.raises(RuntimeError, match="admission-fail"):
        await boundary.execute(Ping(value="ping"))

    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None


@pytest.mark.asyncio
async def test_boundary_resets_identity_after_delegate_exception() -> None:
    identity = _identity_binding()
    boundary = ExecutionBoundary[Ping, Pong](
        FailingPingDelegate(),
        admission_hooks=(RecordingAdmissionHook([], "ok"),),
        identity=identity,
    )

    with pytest.raises(ValueError, match="boom:fail"):
        await boundary.execute(Ping(value="fail"))

    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None


@pytest.mark.asyncio
async def test_nested_boundary_restores_outer_execution_identity() -> None:
    outer = _identity_binding()
    inner = _identity_binding()
    outer_captured: dict[str, ExecutionId] = {}
    inner_captured: dict[str, ExecutionId] = {}

    class CaptureDelegate:
        def __init__(self, bucket: dict[str, ExecutionId], label: str) -> None:
            self._bucket = bucket
            self._label = label

        async def execute(self, request: Ping) -> Pong:
            self._bucket[self._label] = require_active_execution_id()
            return Pong(value=self._label)

    inner_boundary = ExecutionBoundary[Ping, Pong](
        CaptureDelegate(inner_captured, "inner"),
        identity=inner,
    )

    class OuterDelegate:
        async def execute(self, request: Ping) -> Pong:
            outer_captured["outer"] = require_active_execution_id()
            return await inner_boundary.execute(request)

    outer_boundary = ExecutionBoundary[Ping, Pong](
        OuterDelegate(),
        identity=outer,
    )

    await outer_boundary.execute(Ping(value="nested"))

    assert outer_captured["outer"] == outer.execution_id
    assert inner_captured["inner"] == inner.execution_id
    assert outer_captured["outer"] != inner_captured["inner"]
    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None


@pytest.mark.asyncio
async def test_boundary_without_identity_preserves_legacy_behavior() -> None:
    delegate = CountingPingDelegate(Pong(value="pong"))
    boundary = ExecutionBoundary[Ping, Pong](delegate)

    await boundary.execute(Ping(value="ping"))

    assert delegate.call_count == 1
    assert peek_active_execution_identity() is None


def test_execution_identity_binding_not_exported_from_package_root() -> None:
    assert "ExecutionIdentityBinding" not in execution_public_api


@pytest.mark.asyncio
async def test_parallel_boundaries_isolate_execution_ids() -> None:
    identity_a = _identity_binding()
    identity_b = _identity_binding()
    captured: dict[str, ExecutionId] = {}

    class CaptureDelegate:
        def __init__(self, label: str) -> None:
            self._label = label

        async def execute(self, request: Ping) -> Pong:
            await asyncio.sleep(0.05)
            captured[self._label] = require_active_execution_id()
            return Pong(value=self._label)

    async def run_boundary(identity: ExecutionIdentityBinding, label: str) -> None:
        boundary = ExecutionBoundary[Ping, Pong](
            CaptureDelegate(label),
            identity=identity,
        )
        await boundary.execute(Ping(value=label))

    await asyncio.gather(
        run_boundary(identity_a, "a"),
        run_boundary(identity_b, "b"),
    )

    assert captured["a"] == identity_a.execution_id
    assert captured["b"] == identity_b.execution_id
    assert captured["a"] != captured["b"]
    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None


def test_execution_admission_hook_is_generic_protocol() -> None:
    hook: ExecutionAdmissionHook[Ping] = RecordingAdmissionHook([], "typed")
    assert hook is not None


def test_execution_admission_hook_not_exported_from_package_root() -> None:
    assert "ExecutionAdmissionHook" not in execution_public_api


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

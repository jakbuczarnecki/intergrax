# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
    require_active_execution_id,
    require_active_execution_identity,
    validate_execution_id,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.execution import (
    Execution,
    ExecutionCapability,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    RootExecutionOptions,
)
from intergrax.runtime.execution.active_execution_budget import peek_active_execution_budget
from intergrax.runtime.execution.boundary import ExecutionBoundary
from intergrax.runtime.execution.inference import InferenceExecutor
from intergrax.runtime.execution.runtime import ExecutionRuntime
from intergrax.runtime.execution.strategy import StrategyResolver
from intergrax.runtime.execution.strategy_router import StrategyExecutionRouter
from intergrax.runtime.governance.active_execution_authority import require_active_execution_authority
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass(frozen=True)
class Ping:
    value: str


@dataclass(frozen=True)
class Pong:
    value: str


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    risk: str


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


class StructuredProbeAdapter:
    provider = "probe"
    model = "probe"

    def __init__(self, parsed: RiskAssessment) -> None:
        self._parsed = parsed
        self.probe: dict[str, object] = {}

    @property
    def context_window_tokens(self) -> int:
        return 4096

    def supports_structured_output(self) -> bool:
        return True

    def generate_messages(self, messages, **kwargs):
        raise AssertionError("generate_messages must not be called")

    def generate_structured(self, messages, output_model, **kwargs):
        from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
        from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult

        self.probe["run_id_ctx"], self.probe["attempt_id_ctx"] = require_active_execution_identity()
        self.probe["execution_id"] = require_active_execution_id()
        self.probe["authority"] = require_active_execution_authority()
        self.probe["budget"] = peek_active_execution_budget()
        return LLMStructuredResult(parsed=self._parsed, response=build_adapter_response(content=""))

    def generate_with_tools(self, messages, tools_schema, **kwargs):
        raise AssertionError("generate_with_tools must not be called")


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


def _root_options(
    *,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
    tenant_id: str | None = None,
    authority: ParentExecutionAuthority | None = None,
) -> RootExecutionOptions:
    return RootExecutionOptions(
        authority=authority or ParentExecutionAuthority.unrestricted_root(),
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id=tenant_id,
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
    options = _root_options()

    await execution.execute(request, options=options)

    assert delegate.call_count == 1
    assert delegate.last_request == request


@pytest.mark.asyncio
async def test_facade_returns_exact_boundary_result() -> None:
    expected = Pong(value="exact")
    execution = _ping_execution(expected)

    result = await execution.execute(Ping(value="ping"), options=_root_options())

    assert result is expected


@pytest.mark.asyncio
async def test_facade_propagates_boundary_exception_unchanged() -> None:
    boundary = ExecutionBoundary[Ping, Pong](FailingPingDelegate())
    runtime = ExecutionRuntime[Ping, Pong](boundary)
    execution = Execution[Ping, Pong](runtime)

    with pytest.raises(ValueError, match="boom:fail"):
        await execution.execute(Ping(value="fail"), options=_root_options())


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
        await execution.execute(Ping(value="once"), options=_root_options())

    assert delegate.call_count == 1


@pytest.mark.asyncio
async def test_facade_works_with_non_task_typed_dtos() -> None:
    execution = _ping_execution(Pong(value="typed-pong"))

    result = await execution.execute(Ping(value="typed-ping"), options=_root_options())

    assert result == Pong(value="typed-pong")


@pytest.mark.asyncio
async def test_task_typed_boundary_composition() -> None:
    task = _minimal_task()
    expected = _minimal_task_result()
    runner = FakeTaskRunner(expected)
    delegate = TaskExecutionDelegate(runner)
    runtime = ExecutionRuntime[Task, TaskResult](delegate)
    execution = Execution[Task, TaskResult](runtime)

    result = await execution.execute(task, options=_root_options())

    assert runner.call_count == 1
    assert runner.last_task is task
    assert result is expected


@pytest.mark.asyncio
async def test_facade_instances_do_not_share_mutable_state() -> None:
    delegate_a = CountingPingDelegate(Pong(value="a"))
    delegate_b = CountingPingDelegate(Pong(value="b"))
    execution_a = Execution[Ping, Pong](ExecutionRuntime[Ping, Pong](delegate_a))
    execution_b = Execution[Ping, Pong](ExecutionRuntime[Ping, Pong](delegate_b))
    options = _root_options()

    await execution_a.execute(Ping(value="a"), options=options)
    await execution_b.execute(Ping(value="b"), options=options)

    assert delegate_a.call_count == 1
    assert delegate_b.call_count == 1
    assert delegate_a.last_request == Ping(value="a")
    assert delegate_b.last_request == Ping(value="b")


@pytest.mark.asyncio
async def test_facade_mints_platform_execution_id_not_supplied_by_caller() -> None:
    parsed = RiskAssessment(risk="low")
    adapter = StructuredProbeAdapter(parsed)
    router = StrategyExecutionRouter[
        tuple[ChatMessage, ...],
        RiskAssessment,
        ExecutionResult[RiskAssessment],
    ](inference_executor=InferenceExecutor(adapter))
    runtime = ExecutionRuntime[
        ExecutionRequest[tuple[ChatMessage, ...], RiskAssessment],
        ExecutionResult[RiskAssessment],
    ](router)
    execution = Execution(runtime)
    request = ExecutionRequest(
        input=(ChatMessage(role="user", content="probe"),),
        output_type=RiskAssessment,
    )

    await execution.execute(request, options=_root_options())

    execution_id = adapter.probe["execution_id"]
    assert validate_execution_id(execution_id)


@pytest.mark.asyncio
async def test_facade_two_invocations_same_run_attempt_get_distinct_execution_ids() -> None:
    parsed = RiskAssessment(risk="low")
    seen: list[object] = []

    class CapturingAdapter(StructuredProbeAdapter):
        def generate_structured(self, messages, output_model, **kwargs):
            seen.append(require_active_execution_id())
            return super().generate_structured(messages, output_model, **kwargs)

    capturing_adapter = CapturingAdapter(parsed)
    router = StrategyExecutionRouter[
        tuple[ChatMessage, ...],
        RiskAssessment,
        ExecutionResult[RiskAssessment],
    ](inference_executor=InferenceExecutor(capturing_adapter))
    runtime = ExecutionRuntime[
        ExecutionRequest[tuple[ChatMessage, ...], RiskAssessment],
        ExecutionResult[RiskAssessment],
    ](router)
    execution = Execution(runtime)
    request = ExecutionRequest(
        input=(ChatMessage(role="user", content="probe"),),
        output_type=RiskAssessment,
    )
    options = _root_options(run_id=mint_run_id(), attempt_id=mint_attempt_id())

    await execution.execute(request, options=options)
    await execution.execute(request, options=options)

    assert len(seen) == 2
    assert seen[0] != seen[1]
    assert validate_execution_id(seen[0])
    assert validate_execution_id(seen[1])


@pytest.mark.asyncio
async def test_facade_root_budget_and_authority_visible_before_strategy() -> None:
    parsed = RiskAssessment(risk="low")
    adapter = StructuredProbeAdapter(parsed)
    authority = ParentExecutionAuthority.unrestricted_root()
    router = StrategyExecutionRouter[
        tuple[ChatMessage, ...],
        RiskAssessment,
        ExecutionResult[RiskAssessment],
    ](inference_executor=InferenceExecutor(adapter))
    runtime = ExecutionRuntime[
        ExecutionRequest[tuple[ChatMessage, ...], RiskAssessment],
        ExecutionResult[RiskAssessment],
    ](router, run_budget=RunBudget(max_total_tokens=42))
    execution = Execution(runtime)
    request = ExecutionRequest(
        input=(ChatMessage(role="user", content="probe"),),
        output_type=RiskAssessment,
    )

    assert StrategyResolver().resolve(request) is not None

    result = await execution.execute(
        request,
        options=_root_options(tenant_id="tenant-1", authority=authority),
    )

    assert result.status is ExecutionStatus.COMPLETED
    assert adapter.probe["authority"] is authority
    budget = adapter.probe["budget"]
    assert budget is not None
    assert budget.execution_id == adapter.probe["execution_id"]


def test_package_root_exports_execution() -> None:
    from intergrax.runtime.execution import Execution as ExportedExecution
    from intergrax.runtime.execution import RootExecutionOptions as ExportedRootExecutionOptions

    assert ExportedExecution is Execution
    assert ExportedRootExecutionOptions is RootExecutionOptions

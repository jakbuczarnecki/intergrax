# © Artur Czarnecki. All rights reserved.

"""UE-10R1 — canonical root execution runtime tests."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    mint_task_id,
    peek_active_execution_id,
    peek_active_execution_identity,
    peek_active_parent_execution_id,
    require_active_execution_id,
    require_active_execution_identity,
    validate_execution_id,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.execution.active_execution_budget import peek_active_execution_budget
from intergrax.runtime.execution.agentic import AgentExecutor
from intergrax.runtime.execution.facade import Execution
from intergrax.runtime.execution.inference import InferenceExecutor
from intergrax.runtime.execution.orchestration import (
    OrchestrationExecutor,
    TaskBoundOrchestrationDelegate,
    execute_root_task,
    resolve_root_task_identity,
)
from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.result import ExecutionResult, ExecutionStatus
from intergrax.runtime.execution.runtime import (
    ExecutionRuntime,
    RootExecutionContext,
    RootExecutionOptions,
    RootTaskIdentity,
    resolve_root_execution_context,
)
from intergrax.runtime.execution.strategy_router import StrategyExecutionRouter
from intergrax.runtime.execution.task_adapter import TaskExecutionInput, execution_request_from_task
from intergrax.runtime.governance.active_execution_authority import require_active_execution_authority
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class RiskAssessment:
    risk: str


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
        self.probe["adapter_run_id"] = kwargs.get("run_id")
        self.probe["run_id_ctx"], self.probe["attempt_id_ctx"] = require_active_execution_identity()
        self.probe["execution_id"] = require_active_execution_id()
        self.probe["authority"] = require_active_execution_authority()
        budget = peek_active_execution_budget()
        self.probe["budget"] = budget
        from intergrax.llm_adapters.contracts.llm_adapter import LLMStructuredResult
        from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
        from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
        from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response

        return LLMStructuredResult(parsed=self._parsed, response=build_adapter_response(content=""))

    def generate_with_tools(self, messages, tools_schema, **kwargs):
        raise AssertionError("generate_with_tools must not be called")


class AgentEngineProbe:
    async def run_with_result(self, request: RuntimeRequest) -> AgentExecutionResult:
        probe = {
            "run_id_ctx": require_active_execution_identity(),
            "execution_id": require_active_execution_id(),
            "authority": require_active_execution_authority(),
            "budget": peek_active_execution_budget(),
            "request_run_id": request.run_id,
        }
        self.last_probe = probe
        return AgentExecutionResult(
            agent_id="test-agent",
            run_id=request.run_id,
            status=AgentExecutionStatus.COMPLETED,
        )


def _root_options(
    *,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
    tenant_id: str | None = None,
) -> RootExecutionOptions:
    return RootExecutionOptions(
        authority=ParentExecutionAuthority.unrestricted_root(),
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id=tenant_id,
    )


def _root_context(
    *,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
    tenant_id: str | None = None,
) -> RootExecutionContext:
    return resolve_root_execution_context(_root_options(
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id=tenant_id,
    ))


@pytest.mark.asyncio
async def test_inference_root_runtime_binds_identity_authority_budget() -> None:
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
    ](router, run_budget=RunBudget(max_total_tokens=99))
    execution = Execution(runtime)
    options = _root_options(tenant_id="tenant-1")
    request = ExecutionRequest(
        input=(ChatMessage(role="user", content="probe"),),
        output_type=RiskAssessment,
    )

    result = await execution.execute(request, options=options)

    assert result.status is ExecutionStatus.COMPLETED
    assert str(adapter.probe["run_id_ctx"]) == adapter.probe["adapter_run_id"]
    assert adapter.probe["run_id_ctx"] is not None
    assert adapter.probe["attempt_id_ctx"] is not None
    assert validate_execution_id(adapter.probe["execution_id"])
    assert adapter.probe["authority"] is options.authority
    budget = adapter.probe["budget"]
    assert budget is not None
    assert budget.execution_id == adapter.probe["execution_id"]
    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None


@pytest.mark.asyncio
async def test_agentic_root_runtime_binds_identity_authority_budget() -> None:
    engine = AgentEngineProbe()
    router = StrategyExecutionRouter[
        RuntimeRequest,
        AgentExecutionResult,
        ExecutionResult[AgentExecutionResult],
    ](agent_executor=AgentExecutor(engine))
    runtime = ExecutionRuntime[
        ExecutionRequest[RuntimeRequest, AgentExecutionResult],
        ExecutionResult[AgentExecutionResult],
    ](router, run_budget=RunBudget(max_tool_calls=7))
    execution = Execution(runtime)
    run_id = mint_run_id()
    options = _root_options(run_id=run_id)
    request = ExecutionRequest(
        input=RuntimeRequest(
            agent_id="test-agent",
            user_id="user-1",
            session_id="session-1",
            message="agent",
            task_id=mint_task_id(),
            run_id=run_id,
        ),
        output_type=AgentExecutionResult,
        capabilities=frozenset({ExecutionCapability.AGENT}),
    )

    await execution.execute(request, options=options)

    probe = engine.last_probe
    assert probe["run_id_ctx"][0] == run_id
    assert probe["run_id_ctx"][1] is not None
    assert validate_execution_id(probe["execution_id"])
    assert probe["authority"] == options.authority
    assert probe["budget"] is not None
    assert probe["budget"].execution_id == probe["execution_id"]
    assert probe["request_run_id"] == run_id


@pytest.mark.asyncio
async def test_orchestration_root_runtime_nexus_receives_active_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = AgentRegistry()
    loop = NexusLoop(registry)
    captured: dict[str, object] = {}

    async def _fake_impl(task: Task) -> TaskResult:
        run_id, attempt_id = require_active_execution_identity()
        captured["run_id"] = run_id
        captured["attempt_id"] = attempt_id
        captured["execution_id"] = require_active_execution_id()
        captured["authority"] = require_active_execution_authority()
        captured["budget"] = peek_active_execution_budget()
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "_handle_task_impl", _fake_impl)
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="hello",
        context=TaskContext(capability="echo.basic"),
    )
    identity = resolve_root_task_identity()
    result = await execute_root_task(task, nexus_loop=loop, identity=identity)

    assert result.state is TaskState.COMPLETED
    assert captured["run_id"] == identity.run_id
    assert captured["attempt_id"] == identity.attempt_id
    assert captured["execution_id"] == identity.execution_id
    assert captured["authority"] is not None
    budget = captured["budget"]
    assert budget is not None
    assert budget.execution_id == identity.execution_id
    assert peek_active_execution_identity() is None


@pytest.mark.asyncio
async def test_root_lifecycle_shape_identical_across_strategies() -> None:
    shapes: list[tuple[RunId, AttemptId, ExecutionId, ExecutionId | None]] = []
    contexts: list[RootExecutionContext | RootTaskIdentity] = []

    async def _capture_shape() -> None:
        run_id, attempt_id = require_active_execution_identity()
        shapes.append(
            (
                run_id,
                attempt_id,
                require_active_execution_id(),
                peek_active_parent_execution_id(),
            )
        )

    parsed = RiskAssessment(risk="low")
    adapter = StructuredProbeAdapter(parsed)

    class InferenceProbeExecutor:
        async def execute(
            self,
            request: ExecutionRequest[tuple[ChatMessage, ...], RiskAssessment],
        ) -> ExecutionResult[RiskAssessment]:
            await _capture_shape()
            return ExecutionResult(status=ExecutionStatus.COMPLETED, output=parsed)

    inference_context = _root_context()
    contexts.append(inference_context)
    inference_runtime = ExecutionRuntime(
        StrategyExecutionRouter[
            tuple[ChatMessage, ...],
            RiskAssessment,
            ExecutionResult[RiskAssessment],
        ](inference_executor=InferenceProbeExecutor()),
    )
    await inference_runtime.execute(
        ExecutionRequest(
            input=(ChatMessage(role="user", content="x"),),
            output_type=RiskAssessment,
        ),
        inference_context,
    )

    agent_context = _root_context()
    contexts.append(agent_context)

    class AgentProbeExecutor:
        async def execute(
            self,
            request: ExecutionRequest[RuntimeRequest, AgentExecutionResult],
        ) -> ExecutionResult[AgentExecutionResult]:
            await _capture_shape()
            run_id, _ = require_active_execution_identity()
            del request
            return ExecutionResult(
                status=ExecutionStatus.COMPLETED,
                output=AgentExecutionResult(
                    agent_id="test-agent",
                    run_id=run_id,
                    status=AgentExecutionStatus.COMPLETED,
                ),
            )

    agent_runtime = ExecutionRuntime(
        StrategyExecutionRouter[
            RuntimeRequest,
            AgentExecutionResult,
            ExecutionResult[AgentExecutionResult],
        ](agent_executor=AgentProbeExecutor()),
    )
    await agent_runtime.execute(
        ExecutionRequest(
            input=RuntimeRequest(
                agent_id="test-agent",
                user_id="user-1",
                session_id="session-1",
                message="x",
                task_id=mint_task_id(),
                run_id=agent_context.run_id,
            ),
            output_type=AgentExecutionResult,
            capabilities=frozenset({ExecutionCapability.AGENT}),
        ),
        agent_context,
    )

    loop = NexusLoop(AgentRegistry())
    task = Task(
        task_id=mint_task_id(),
        tenant_id="t1",
        user_id="u1",
        message="x",
        context=TaskContext(),
    )
    orch_identity = resolve_root_task_identity()
    contexts.append(orch_identity)

    async def _orch_capture(task: Task) -> TaskResult:
        await _capture_shape()
        run_id, _ = require_active_execution_identity()
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    loop._handle_task_impl = _orch_capture  # type: ignore[method-assign]
    await execute_root_task(
        task,
        nexus_loop=loop,
        identity=orch_identity,
    )

    assert len(shapes) == 3
    for index, (run_id, attempt_id, execution_id, parent_id) in enumerate(shapes):
        expected_execution_id = contexts[index].execution_id
        assert run_id is not None
        assert attempt_id is not None
        assert execution_id == expected_execution_id
        assert parent_id is None


@pytest.mark.asyncio
async def test_nexus_without_active_identity_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = NexusLoop(AgentRegistry())
    monkeypatch.setattr(
        loop,
        "_handle_task_impl",
        AsyncMock(return_value=TaskResult(task_id=mint_task_id(), state=TaskState.COMPLETED)),
    )
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="fail")

    with pytest.raises(RuntimeError, match="active ExecutionId required"):
        await loop.handle_task(task, run_id=mint_run_id())


@pytest.mark.asyncio
async def test_nexus_without_active_authority_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from intergrax.contracts.execution_identity import bind_active_execution_identity, reset_active_execution_identity
    from intergrax.runtime.execution.active_execution_budget import bind_root_execution_budget, reset_active_execution_budget
    from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger

    loop = NexusLoop(AgentRegistry())
    monkeypatch.setattr(
        loop,
        "_handle_task_impl",
        AsyncMock(return_value=TaskResult(task_id=mint_task_id(), state=TaskState.COMPLETED)),
    )
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="fail")
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    ledger = create_execution_budget_ledger(RunBudget())
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    budget_token = bind_root_execution_budget(execution_id=execution_id, ledger=ledger)
    try:
        with pytest.raises(RuntimeError, match="active execution authority required"):
            await loop.handle_task(task, run_id=run_id, attempt_id=attempt_id)
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_identity(identity_token)


@pytest.mark.asyncio
async def test_nexus_without_active_budget_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from intergrax.contracts.execution_identity import bind_active_execution_identity, reset_active_execution_identity
    from intergrax.runtime.governance.active_execution_authority import bind_active_execution_authority, reset_active_execution_authority

    loop = NexusLoop(AgentRegistry())
    monkeypatch.setattr(
        loop,
        "_handle_task_impl",
        AsyncMock(return_value=TaskResult(task_id=mint_task_id(), state=TaskState.COMPLETED)),
    )
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="fail")
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    authority = ParentExecutionAuthority.unrestricted_root()
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    authority_token = bind_active_execution_authority(authority)
    try:
        with pytest.raises(RuntimeError, match="active execution budget required"):
            await loop.handle_task(task, run_id=run_id, attempt_id=attempt_id)
    finally:
        reset_active_execution_authority(authority_token)
        reset_active_execution_identity(identity_token)


@pytest.mark.asyncio
async def test_resume_root_execution_id_matches_identity_through_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from intergrax.runtime.long_running.execution_tree_checkpoint import (
        minimal_runtime_checkpoint,
    )
    from intergrax.runtime.long_running.models import TaskCheckpoint

    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    historical_root_id = mint_execution_id()
    registry = AgentRegistry()
    loop = NexusLoop(registry)
    captured: dict[str, ExecutionId] = {}

    async def _fake_impl(task: Task) -> TaskResult:
        captured["active_execution_id"] = require_active_execution_id()
        active_run_id, _ = require_active_execution_identity()
        return TaskResult(task_id=task.task_id, run_id=active_run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "_handle_task_impl", _fake_impl)
    task = Task(
        task_id=mint_task_id(),
        tenant_id="t1",
        user_id="u1",
        message="resume",
        context=TaskContext(),
    )
    checkpoint = TaskCheckpoint(
        task_id=task.task_id,
        tenant_id="t1",
        resume_token="rt_resume",
        task_state=TaskState.WAITING_FOR_HUMAN,
        runtime=minimal_runtime_checkpoint(
            task_id=task.task_id,
            run_id=run_id,
            attempt_id=attempt_a1,
            root_execution_id=historical_root_id,
        ),
    )
    identity = resolve_root_task_identity(
        run_id=run_id,
        attempt_id=attempt_a1,
        resume_checkpoint=checkpoint,
    )

    await execute_root_task(
        task,
        nexus_loop=loop,
        identity=identity,
        resume_checkpoint=checkpoint,
    )

    assert historical_root_id != identity.execution_id
    assert task.runtime.orchestration.runtime_checkpoint is not None
    execution_tree = task.runtime.orchestration.runtime_checkpoint.execution_tree
    root_entries = [
        entry
        for entry in execution_tree.entries
        if entry.parent_execution_id is None
    ]
    assert len(root_entries) == 1
    assert root_entries[0].execution_id == identity.execution_id
    assert captured["active_execution_id"] == identity.execution_id

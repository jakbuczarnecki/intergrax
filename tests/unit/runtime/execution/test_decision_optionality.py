# © Artur Czarnecki. All rights reserved.

"""DS-EXEC-00 — prove ordinary Execution works without Decision capability.

Decision integration may exist in Execution (DS-EXEC-01 optional host hook),
but ordinary flows must remain constructible and executable without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.agent_execution_result import (
    AgentExecutionResult,
    AgentExecutionStatus,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    RunId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.runtime.execution import (
    ExecutionCapability,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
)
from intergrax.runtime.execution.agentic import AgentExecutor
from intergrax.runtime.execution.facade import Execution
from intergrax.runtime.execution.inference import InferenceExecutor
from intergrax.runtime.execution.orchestration import (
    execute_root_task,
    resolve_root_task_identity,
)
from intergrax.runtime.execution.request import ExecutionRequest as NeutralExecutionRequest
from intergrax.runtime.execution.runtime import (
    ExecutionRuntime,
    RootExecutionContext,
    RootExecutionOptions,
)
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver
from intergrax.runtime.execution.strategy_router import StrategyExecutionRouter
from intergrax.runtime.execution.task_adapter import TaskExecutionInput, execution_request_from_task
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_STRATEGY_ROUTING_FILES = (
    Path("intergrax/runtime/execution/strategy.py"),
    Path("intergrax/runtime/execution/strategy_router.py"),
)

_GLOBAL_DECISION_DISABLE_WORKAROUND_TOKENS = frozenset(
    {
        "NoDecisionStrategy",
        "NullDecisionStrategy",
        "DisabledDecisionStrategy",
        "BypassDecisionStrategy",
        "DECISION_SYSTEM_ENABLED",
        "decision_enabled",
        "enable_decision_system",
        "disable_decision_system",
    }
)


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    risk: str


class StructuredOptionalityAdapter(LLMAdapter):
    provider = LLMProvider.OLLAMA
    model = "optionality-probe"

    def __init__(self, parsed_output: RiskAssessment) -> None:
        super().__init__()
        self.parsed_output = parsed_output
        self.generate_structured_calls = 0

    @property
    def context_window_tokens(self) -> int:
        return 4096

    def supports_structured_output(self) -> bool:
        return True

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        raise AssertionError("generate_messages must not be called")

    def generate_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools_schema: list,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: str | dict | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        raise AssertionError("generate_with_tools must not be called")

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMStructuredResult[RiskAssessment]:
        self.generate_structured_calls += 1
        require_active_execution_identity()
        require_active_execution_id()
        return LLMStructuredResult(
            parsed=self.parsed_output,
            response=build_adapter_response(content=""),
        )


class RecordingAgentEngine:
    def __init__(self, *, run_id: RunId) -> None:
        self.run_id = run_id
        self.calls: list[RuntimeRequest] = []

    async def run_with_result(
        self,
        request: RuntimeRequest,
    ) -> AgentExecutionResult:
        self.calls.append(request)
        require_active_execution_identity()
        require_active_execution_id()
        return AgentExecutionResult(
            agent_id=request.agent_id,
            run_id=request.run_id,
            status=AgentExecutionStatus.COMPLETED,
            summary="optionality-probe",
        )


def _root_options(*, run_id: RunId | None = None) -> RootExecutionOptions:
    return RootExecutionOptions(
        authority=ParentExecutionAuthority.unrestricted_root(),
        run_id=run_id,
    )


def _inference_execution_stack(
    adapter: StructuredOptionalityAdapter,
) -> Execution[
    ExecutionRequest[tuple[ChatMessage, ...], RiskAssessment],
    ExecutionResult[RiskAssessment],
]:
    router = StrategyExecutionRouter[
        tuple[ChatMessage, ...],
        RiskAssessment,
        ExecutionResult[RiskAssessment],
    ](inference_executor=InferenceExecutor(adapter))
    runtime = ExecutionRuntime[
        ExecutionRequest[tuple[ChatMessage, ...], RiskAssessment],
        ExecutionResult[RiskAssessment],
    ](router)
    return Execution(runtime)


def _agentic_execution_stack(
    engine: RecordingAgentEngine,
) -> Execution[
    ExecutionRequest[RuntimeRequest, AgentExecutionResult],
    ExecutionResult[AgentExecutionResult],
]:
    router = StrategyExecutionRouter[
        RuntimeRequest,
        AgentExecutionResult,
        ExecutionResult[AgentExecutionResult],
    ](agent_executor=AgentExecutor(engine))
    runtime = ExecutionRuntime[
        ExecutionRequest[RuntimeRequest, AgentExecutionResult],
        ExecutionResult[AgentExecutionResult],
    ](router)
    return Execution(runtime)


@pytest.mark.asyncio
async def test_inference_root_runtime_completes_without_decision() -> None:
    expected = RiskAssessment(risk="low")
    adapter = StructuredOptionalityAdapter(parsed_output=expected)
    execution = _inference_execution_stack(adapter)
    request = ExecutionRequest(
        input=(ChatMessage(role="user", content="probe"),),
        output_type=RiskAssessment,
    )

    assert StrategyResolver().resolve(request) is ExecutionStrategy.INFERENCE

    result = await execution.execute(request, options=_root_options())

    assert result.status is ExecutionStatus.COMPLETED
    assert result.output == expected
    assert adapter.generate_structured_calls == 1


@pytest.mark.asyncio
async def test_agentic_root_runtime_completes_without_decision() -> None:
    run_id = mint_run_id()
    runtime_request = RuntimeRequest(
        agent_id="probe-agent",
        user_id="user-1",
        session_id="session-1",
        message="hello",
        task_id=mint_task_id(),
        run_id=run_id,
    )
    request = ExecutionRequest(
        input=runtime_request,
        output_type=AgentExecutionResult,
        capabilities=frozenset({ExecutionCapability.TOOLS}),
    )
    engine = RecordingAgentEngine(run_id=run_id)
    execution = _agentic_execution_stack(engine)

    assert StrategyResolver().resolve(request) is ExecutionStrategy.AGENTIC

    result = await execution.execute(request, options=_root_options(run_id=run_id))

    assert result.status is ExecutionStatus.COMPLETED
    assert result.output.status is AgentExecutionStatus.COMPLETED
    assert len(engine.calls) == 1
    assert engine.calls[0] is runtime_request


@pytest.mark.asyncio
async def test_orchestration_root_runtime_completes_without_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="hello",
        context=TaskContext(capability="echo.basic"),
    )
    expected = TaskResult(task_id=task.task_id, state=TaskState.COMPLETED, answer="ok")
    registry = AgentRegistry()
    loop = NexusLoop(registry)
    request = execution_request_from_task(
        task,
        capabilities=frozenset({ExecutionCapability.ORCHESTRATION}),
        output_type=TaskResult,
    )

    assert StrategyResolver().resolve(request) is ExecutionStrategy.ORCHESTRATION

    async def _fake_handle_task(task_arg: Task) -> TaskResult:
        require_active_execution_identity()
        require_active_execution_id()
        del task_arg
        return expected

    monkeypatch.setattr(loop, "_handle_task_impl", _fake_handle_task)

    identity = resolve_root_task_identity()
    result = await execute_root_task(task, nexus_loop=loop, identity=identity)

    assert result is expected


def test_root_execution_options_constructible_without_decision_config() -> None:
    options = RootExecutionOptions(
        authority=ParentExecutionAuthority.unrestricted_root(),
    )

    assert options.authority is not None
    assert options.run_id is None
    assert options.attempt_id is None
    assert options.tenant_id is None


def test_ordinary_execution_request_constructible_without_decision_config() -> None:
    request = ExecutionRequest(
        input=(ChatMessage(role="user", content="probe"),),
        output_type=RiskAssessment,
    )
    neutral_request = NeutralExecutionRequest(
        input=TaskExecutionInput(message="probe"),
        output_type=TaskResult,
        capabilities=frozenset({ExecutionCapability.ORCHESTRATION}),
    )
    run_id = mint_run_id()
    context = RootExecutionContext(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        authority=ParentExecutionAuthority.unrestricted_root(),
    )

    assert request.input is not None
    assert request.output_type is RiskAssessment
    assert neutral_request.capabilities == frozenset({ExecutionCapability.ORCHESTRATION})
    assert context.run_id == run_id


def test_strategy_routing_has_no_global_decision_disable_workaround() -> None:
    for path in _STRATEGY_ROUTING_FILES:
        source = path.read_text(encoding="utf-8")
        for token in _GLOBAL_DECISION_DISABLE_WORKAROUND_TOKENS:
            assert token not in source, (
                f"{path} introduces global Decision disable workaround: {token}"
            )


def test_strategy_resolver_is_orthogonal_to_decision_strategy() -> None:
    strategy_source = Path("intergrax/runtime/execution/strategy.py").read_text(
        encoding="utf-8"
    )
    router_source = Path("intergrax/runtime/execution/strategy_router.py").read_text(
        encoding="utf-8"
    )

    for token in (
        "DecisionStrategy",
        "DecisionStrategyKind",
        "Single Model",
        "Rule-Based",
        "Hybrid",
        "Council",
    ):
        assert token not in strategy_source
        assert token not in router_source

    resolver = StrategyResolver()
    inference_request = ExecutionRequest(
        input=(ChatMessage(role="user", content="probe"),),
        output_type=RiskAssessment,
    )
    agentic_request = ExecutionRequest(
        input=RuntimeRequest(
            agent_id="probe",
            user_id="user",
            session_id="session",
            message="probe",
            task_id=mint_task_id(),
            run_id=mint_run_id(),
        ),
        output_type=AgentExecutionResult,
        capabilities=frozenset({ExecutionCapability.AGENT}),
    )
    orchestration_request = ExecutionRequest(
        input=TaskExecutionInput(message="probe"),
        output_type=TaskResult,
        capabilities=frozenset({ExecutionCapability.ORCHESTRATION}),
    )

    assert resolver.resolve(inference_request) is ExecutionStrategy.INFERENCE
    assert resolver.resolve(agentic_request) is ExecutionStrategy.AGENTIC
    assert resolver.resolve(orchestration_request) is ExecutionStrategy.ORCHESTRATION

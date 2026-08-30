# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.contracts.agent_execution_result import (
    AgentExecutionResult,
    AgentExecutionStatus,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.runtime.execution import ExecutionCapability, ExecutionRequest, ExecutionStatus
from intergrax.runtime.execution.agentic import AgentExecutor
from intergrax.runtime.execution.inference import InferenceExecutor
from intergrax.runtime.execution.orchestration import (
    OrchestrationExecutor,
    TaskBoundOrchestrationDelegate,
)
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver
from intergrax.runtime.execution.strategy_router import StrategyExecutionRouter
from intergrax.runtime.execution.task_adapter import execution_request_from_task
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState

pytestmark = pytest.mark.unit


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    risk: str


class StructuredInferenceAdapter(LLMAdapter):
    provider = "test"
    model = "test"

    def __init__(self) -> None:
        super().__init__()
        self.generate_structured_calls = 0

    @property
    def context_window_tokens(self) -> int:
        return 8192

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
        return LLMStructuredResult(
            parsed=RiskAssessment(risk="low"),
            response=build_adapter_response(content=""),
        )


class RecordingAgentEngine:
    calls = 0

    async def run_with_result(self, request: RuntimeRequest) -> AgentExecutionResult:
        RecordingAgentEngine.calls += 1
        return AgentExecutionResult(
            agent_id=request.agent_id,
            run_id=request.run_id,
            status=AgentExecutionStatus.COMPLETED,
            summary="ok",
        )


def _identity_token():
    return bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


@pytest.fixture(autouse=True)
def _reset_agent_engine_calls() -> None:
    RecordingAgentEngine.calls = 0


@pytest.mark.asyncio
async def test_inference_router_delegates_only_to_inference_executor() -> None:
    adapter = StructuredInferenceAdapter()
    inference = InferenceExecutor[RiskAssessment](adapter)
    agent = AgentExecutor(RecordingAgentEngine())
    nexus_loop = MagicMock()
    nexus_loop.handle_task = AsyncMock()
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="hello",
        context=TaskContext(),
    )
    router = StrategyExecutionRouter[
        tuple[ChatMessage, ...],
        RiskAssessment,
        object,
    ](
        inference_executor=inference,
        agent_executor=agent,
        orchestration_executor=TaskBoundOrchestrationDelegate(
            task,
            OrchestrationExecutor(nexus_loop),
        ),
    )
    request = ExecutionRequest(
        input=(ChatMessage(role="user", content="assess"),),
        output_type=RiskAssessment,
        capabilities=frozenset(),
    )

    assert StrategyResolver().resolve(request) is ExecutionStrategy.INFERENCE

    token = _identity_token()
    try:
        result = await router.execute(request)
    finally:
        reset_active_execution_identity(token)

    assert result.status is ExecutionStatus.COMPLETED
    assert adapter.generate_structured_calls == 1
    assert RecordingAgentEngine.calls == 0
    nexus_loop.handle_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_agentic_router_delegates_only_to_agent_executor() -> None:
    adapter = StructuredInferenceAdapter()
    inference = InferenceExecutor[RiskAssessment](adapter)
    agent = AgentExecutor(RecordingAgentEngine())
    nexus_loop = MagicMock()
    nexus_loop.handle_task = AsyncMock()
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="hello",
        context=TaskContext(),
    )
    router = StrategyExecutionRouter[
        RuntimeRequest,
        AgentExecutionResult,
        object,
    ](
        inference_executor=inference,
        agent_executor=agent,
        orchestration_executor=TaskBoundOrchestrationDelegate(
            task,
            OrchestrationExecutor(nexus_loop),
        ),
    )
    run_id = mint_run_id()
    runtime_request = RuntimeRequest(
        agent_id="agent-1",
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

    assert StrategyResolver().resolve(request) is ExecutionStrategy.AGENTIC

    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    try:
        result = await router.execute(request)
    finally:
        reset_active_execution_identity(token)

    assert result.status is ExecutionStatus.COMPLETED
    assert adapter.generate_structured_calls == 0
    assert RecordingAgentEngine.calls == 1
    nexus_loop.handle_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_orchestration_router_delegates_only_to_orchestration_executor() -> None:
    adapter = StructuredInferenceAdapter()
    inference = InferenceExecutor[RiskAssessment](adapter)
    agent = AgentExecutor(RecordingAgentEngine())
    nexus_loop = MagicMock()
    nexus_loop.handle_task = AsyncMock(
        return_value=TaskResult(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
        )
    )
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="hello",
        context=TaskContext(),
    )
    router = StrategyExecutionRouter[
        object,
        TaskResult,
        TaskResult,
    ](
        inference_executor=inference,
        agent_executor=agent,
        orchestration_executor=TaskBoundOrchestrationDelegate(
            task,
            OrchestrationExecutor(nexus_loop),
        ),
    )
    request = execution_request_from_task(
        task,
        capabilities=frozenset({ExecutionCapability.ORCHESTRATION}),
        output_type=TaskResult,
    )

    assert StrategyResolver().resolve(request) is ExecutionStrategy.ORCHESTRATION

    token = _identity_token()
    try:
        await router.execute(request)
    finally:
        reset_active_execution_identity(token)

    assert adapter.generate_structured_calls == 0
    assert RecordingAgentEngine.calls == 0
    nexus_loop.handle_task.assert_awaited_once()

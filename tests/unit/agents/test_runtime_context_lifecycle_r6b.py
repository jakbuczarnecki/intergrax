# © Artur Czarnecki. All rights reserved.

"""TOOLS-SIDE-EFFECT-SAFETY R6B — RuntimeContext lifecycle ownership."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents.authoring.acp_uaep_shim import close_acp_catalog_exec_ctx
from intergrax.agents.authoring.base import IntergraxAgent
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.agents.uaep import UAEPExecutor
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_run import AgentRunRequest, AgentRunResult, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunStatus, TerminalReason
from intergrax.contracts.agent_run_trace import AgentRunTrace
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


def _runtime_context() -> RuntimeContext:
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(fixed_text="stub"),
        enable_rag=False,
        production_mode=False,
        tenant_id="t1",
    )
    return RuntimeContext.build(
        config=config,
        session_manager=build_in_memory_session_manager(),
    )


def _uaep_request(**metadata: object) -> RuntimeRequest:
    return RuntimeRequest(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        agent_id="uaep-lifecycle",
        message="hi",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        metadata=dict(metadata),
    )


def _bind_identity() -> object:
    return bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


class _UaepStubAgent(Agent):
    def __init__(self, *, fail_in_steps: bool = False) -> None:
        self.fail_in_steps = fail_in_steps
        self.close_count = 0

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="uaep-lifecycle",
            name="UAEP Lifecycle",
            description="lifecycle probe",
            capabilities=["stub.basic"],
            max_steps=3,
        )

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        _ = request
        runtime_context = _runtime_context()
        original_close = runtime_context.close

        def tracked_close() -> None:
            self.close_count += 1
            original_close()

        runtime_context.close = tracked_close  # type: ignore[method-assign]
        return runtime_context

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        if self.fail_in_steps:
            raise RuntimeError("step resolution failed")
        _ = context
        return [AgentStep(step_id="s1", step_name="only", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        _ = ctx
        return StepOutput(step_id=step.step_id, summary="ok")

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(type=AgentDecisionType.COMPLETE, reason="done")


class _CancelUaepAgent(_UaepStubAgent):
    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        _ = step, ctx
        raise AssertionError("step must not run when cancelled")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_uaep_success_closes_runtime_context_once() -> None:
    agent = _UaepStubAgent()
    token = _bind_identity()
    try:
        answer, validation, governance = await UAEPExecutor().execute(agent, _uaep_request())
    finally:
        reset_active_execution_identity(token)

    assert validation.valid
    assert answer.answer == "ok"
    assert agent.close_count == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_uaep_failure_after_build_context_closes_once() -> None:
    agent = _UaepStubAgent(fail_in_steps=True)
    token = _bind_identity()
    try:
        with pytest.raises(RuntimeError, match="step resolution failed"):
            await UAEPExecutor().execute(agent, _uaep_request())
    finally:
        reset_active_execution_identity(token)

    assert agent.close_count == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_uaep_early_return_cancellation_closes_context() -> None:
    agent = _CancelUaepAgent()
    token = _bind_identity()
    try:
        answer, validation, governance = await UAEPExecutor().execute(
            agent,
            _uaep_request(cancellation_requested=True),
        )
    finally:
        reset_active_execution_identity(token)

    assert not validation.valid
    assert "task_cancelled" in validation.errors
    assert governance is None
    assert agent.close_count == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_uaep_execute_return_has_no_runtime_context() -> None:
    agent = _UaepStubAgent()
    token = _bind_identity()
    try:
        result = await UAEPExecutor().execute(agent, _uaep_request())
    finally:
        reset_active_execution_identity(token)

    assert len(result) == 3
    assert isinstance(result[0], RuntimeAnswer)
    assert isinstance(result[1], ValidationResult)


class _CounterAgent(IntergraxAgent):
    contract_id = "counter-r6b"
    capabilities = ("demo.counter",)
    agent_name = "Counter"
    agent_description = "Counts steps"
    max_steps = 5

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        return _runtime_context()

    async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
        if step_ctx.step_index >= 1:
            return StepOutcome.complete(
                output={"steps": step_ctx.step_index + 1},
                terminal_reason=TerminalReason.GOAL_MET,
            )
        return StepOutcome.continue_with(state_delta={"phase": f"p{step_ctx.step_index}"})


@pytest.mark.asyncio
@pytest.mark.unit
async def test_agent_engine_acp_branch_does_not_build_post_hoc_context() -> None:
    agent = _CounterAgent()
    run_id = mint_run_id()
    task_id = mint_task_id()
    request = RuntimeRequest(
        agent_id=agent.contract_id,
        tenant_id="tenant-a",
        user_id="user-1",
        session_id="sess-1",
        message="hello",
        task_id=task_id,
        run_id=run_id,
        metadata={AcpMetadataKey.SESSION_ENABLED: True},
    )
    build_context_mock = MagicMock(side_effect=AssertionError("post-hoc build_context"))
    acp_result = AgentRunResult(
        run_id=str(run_id),
        trace_id=str(run_id),
        status=AgentRunStatus.SUCCEEDED,
        terminal_reason=TerminalReason.GOAL_MET,
        trace=AgentRunTrace(run_id=str(run_id)),
    )
    with patch.object(agent, "build_context", build_context_mock):
        with patch.object(agent, "run", AsyncMock(return_value=acp_result)):
            answer = await AgentEngine.run_agent(agent, request)

    build_context_mock.assert_not_called()


@pytest.mark.unit
def test_close_acp_catalog_exec_ctx_closes_runtime_context_once() -> None:
    run_id = mint_run_id()
    task_id = mint_task_id()
    runtime_context = _runtime_context()
    close_count = 0
    original_close = runtime_context.close

    def tracked_close() -> None:
        nonlocal close_count
        close_count += 1
        original_close()

    runtime_context.close = tracked_close  # type: ignore[method-assign]
    exec_ctx = RuntimeExecutionContext(
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        agent_id="counter-r6b",
    )
    exec_ctx.metadata["runtime_state"] = RuntimeState(
        context=runtime_context,
        request=RuntimeRequest(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            agent_id="uaep-lifecycle",
            message="hi",
            task_id=task_id,
            run_id=run_id,
        ),
        run_id=run_id,
    )
    step_ctx = AgentStepContext(
        step_index=0,
        run_id=run_id,
        agent_id="counter-r6b",
        contract_id="counter-r6b",
        metadata={"uaep_exec_ctx": exec_ctx},
    )

    close_acp_catalog_exec_ctx(step_ctx)

    assert close_count == 1
    assert "uaep_exec_ctx" not in step_ctx.metadata


@pytest.mark.unit
def test_close_acp_catalog_exec_ctx_noop_when_missing() -> None:
    step_ctx = AgentStepContext(
        step_index=0,
        run_id=mint_run_id(),
        agent_id="counter-r6b",
        contract_id="counter-r6b",
        metadata={},
    )
    close_acp_catalog_exec_ctx(step_ctx)
    assert "uaep_exec_ctx" not in step_ctx.metadata


@pytest.mark.unit
def test_runtime_context_closes_owned_tool_invoker() -> None:
    runtime_context = _runtime_context()
    invoker = runtime_context.config.tool_invoker
    assert invoker is not None
    runtime_context.close()
    assert invoker._execution_pool_closed


@pytest.mark.unit
def test_host_managed_tool_invoker_survives_runtime_context_close() -> None:
    from intergrax.agents.authoring.acp_uaep_shim import apply_host_tool_invoker_to_runtime_context
    from intergrax.agents.persistence.catalog_declarative_invoker import (
        build_catalog_declarative_invoker_from_registry,
    )
    from intergrax.agents.persistence.tool_invoker_wiring import attach_declarative_tool_invoker
    from intergrax.tools.registry import ToolRegistry
    from intergrax.tools.registry.bootstrap import register_default_tools

    register_default_tools()
    catalog = build_catalog_declarative_invoker_from_registry(ToolRegistry())
    host_invoker = catalog.tool_invoker
    runtime_context = _runtime_context()
    metadata = attach_declarative_tool_invoker({}, catalog)
    apply_host_tool_invoker_to_runtime_context(runtime_context, metadata)
    assert runtime_context.tool_invoker_close_on_context_close is False
    assert runtime_context.config.tool_invoker is host_invoker
    runtime_context.close()
    assert not host_invoker._execution_pool_closed

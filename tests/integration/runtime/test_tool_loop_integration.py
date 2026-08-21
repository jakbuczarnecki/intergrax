# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-6 — bounded multi-iteration tool loop integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.nexus.budget.budget_enforcer import BudgetExceededError
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.config_types import ToolsContextScope
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.runtime.nexus.tools.tool_loop import run_bounded_tool_loop
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.nexus.tools.tool_planning_service import ToolPlanningService
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.tool_executor import ToolHandler
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager, tools_agent_make_contract

pytestmark = [pytest.mark.integration, pytest.mark.gate]

_BEYOND_PREVIEW_MARKER = "ENG1_TAIL_MARKER"
_PREVIEW_BOUND = 400


class _InA(BaseModel):
    value: int = 1


class _OutA(BaseModel):
    result: int = 0


class _HandlerA(ToolHandler[_InA, _OutA]):
    def execute(self, request: ToolExecutionRequest[_InA]) -> _OutA:
        return _OutA(result=request.input.value)


class _LongOut(BaseModel):
    padding: str
    decision_token: str


class _LongIn(BaseModel):
    pass


class _LongHandler(ToolHandler[_LongIn, _LongOut]):
    def execute(self, request: ToolExecutionRequest[_LongIn]) -> _LongOut:
        _ = request
        return _LongOut(padding="x" * 500, decision_token=_BEYOND_PREVIEW_MARKER)


class _FailIn(BaseModel):
    pass


class _FailOut(BaseModel):
    ok: bool = True


class _FailHandler(ToolHandler[_FailIn, _FailOut]):
    def execute(self, request: ToolExecutionRequest[_FailIn]) -> _FailOut:
        _ = request
        raise RuntimeError("x" * 400 + _BEYOND_PREVIEW_MARKER)


class _TwoRoundLLM(FakeLLMAdapter):
    def __init__(self) -> None:
        super().__init__(fixed_text="")
        self._round = 0

    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(self, messages, tools_schema, **kwargs):  # type: ignore[no-untyped-def]
        _ = messages, tools_schema, kwargs
        self._round += 1
        if self._round == 1:
            return LLMAdapterResponse(
                content="",
                tool_calls=(
                    LLMToolCall.from_openai_shape(
                        call_id="tc-1",
                        name="alpha.tool",
                        arguments={"value": 7},
                    ),
                ),
            )
        return LLMAdapterResponse(content="done", tool_calls=())


def _runtime_state(llm: FakeLLMAdapter) -> RuntimeState:
    config = RuntimeConfig(
        llm_adapter=llm,
        production_mode=False,
        enable_rag=False,
        enable_websearch=False,
        tools_context_scope=ToolsContextScope.CURRENT_MESSAGE_ONLY,
        max_tool_iterations=2,
    )
    ctx = RuntimeContext(
        config=config,
        session_manager=build_in_memory_session_manager(),
        prompt_registry=MagicMock(),
    )
    return RuntimeState(
        context=ctx,
        request=RuntimeRequest(
            agent_id="agent-1",
            user_id="user-1",
            session_id="session-1",
            tenant_id="tenant-1",
            task_id=TaskId("task_00000000000000000000000000000001"),
            run_id=RunId("run_00000000000000000000000000000001"),
            message="use tool",
        ),
        run_id="run_00000000000000000000000000000002",
        messages_for_llm=[ChatMessage(role="user", content="use tool")],
    )


def test_bounded_tool_loop_two_iterations_native_messages() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _OutA),
        _HandlerA(),
    )
    llm = _TwoRoundLLM()
    state = _runtime_state(llm)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = ToolPlanningService(llm=llm, tools=registry)

    result = run_bounded_tool_loop(
        state=state,
        invoker=invoker,
        tool_planner=planner,
        planner_input=[ChatMessage(role="user", content="use tool")],
        allowed_tool_ids=("alpha.tool",),
        max_iterations=2,
    )

    assert result.loop_iterations == 2
    assert len(result.tool_traces) == 1
    assert result.tool_traces[0].tool_name == "alpha.tool"
    assert result.used_native_tool_messages is True
    assert any(msg.role == "tool" for msg in result.appended_messages)
    assert llm._round == 2
    assert result.stop_reason == "planner_final_answer"


class _EmptyPlanLLM(FakeLLMAdapter):
    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(self, messages, tools_schema, **kwargs):  # type: ignore[no-untyped-def]
        _ = messages, tools_schema, kwargs
        return LLMAdapterResponse(content="", tool_calls=())


class _AlwaysToolLLM(FakeLLMAdapter):
    def __init__(self) -> None:
        super().__init__(fixed_text="")
        self._round = 0

    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(self, messages, tools_schema, **kwargs):  # type: ignore[no-untyped-def]
        _ = messages, tools_schema, kwargs
        self._round += 1
        return LLMAdapterResponse(
            content="",
            tool_calls=(
                LLMToolCall.from_openai_shape(
                    call_id=f"tc-{self._round}",
                    name="alpha.tool",
                    arguments={"value": self._round},
                ),
            ),
        )


class _PlannerExplodedError(RuntimeError):
    pass


class _FailAfterOneRoundLLM(FakeLLMAdapter):
    def __init__(self) -> None:
        super().__init__(fixed_text="")
        self._round = 0

    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(self, messages, tools_schema, **kwargs):  # type: ignore[no-untyped-def]
        _ = messages, tools_schema, kwargs
        self._round += 1
        if self._round == 1:
            return LLMAdapterResponse(
                content="",
                tool_calls=(
                    LLMToolCall.from_openai_shape(
                        call_id="tc-1",
                        name="alpha.tool",
                        arguments={"value": 1},
                    ),
                ),
            )
        raise _PlannerExplodedError("planner exploded deterministically")


def test_bounded_tool_loop_empty_tool_calls_stop_reason() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _OutA),
        _HandlerA(),
    )
    llm = _EmptyPlanLLM()
    state = _runtime_state(llm)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = ToolPlanningService(llm=llm, tools=registry)

    result = run_bounded_tool_loop(
        state=state,
        invoker=invoker,
        tool_planner=planner,
        planner_input=[ChatMessage(role="user", content="use tool")],
        allowed_tool_ids=("alpha.tool",),
        max_iterations=3,
    )

    assert result.loop_iterations == 1
    assert result.tool_traces == []
    assert result.stop_reason == "empty_tool_calls"


def test_bounded_tool_loop_max_iterations_stop_reason() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _OutA),
        _HandlerA(),
    )
    llm = _AlwaysToolLLM()
    state = _runtime_state(llm)
    state.context.config.max_tool_iterations = 3
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = ToolPlanningService(llm=llm, tools=registry)

    result = run_bounded_tool_loop(
        state=state,
        invoker=invoker,
        tool_planner=planner,
        planner_input=[ChatMessage(role="user", content="use tool")],
        allowed_tool_ids=("alpha.tool",),
        max_iterations=3,
    )

    assert result.loop_iterations == 3
    assert len(result.tool_traces) == 3
    assert result.stop_reason == "max_iterations"
    assert llm._round == 3


def test_bounded_tool_loop_planner_failure_propagates() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _OutA),
        _HandlerA(),
    )
    llm = _FailAfterOneRoundLLM()
    state = _runtime_state(llm)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = ToolPlanningService(llm=llm, tools=registry)

    with pytest.raises(_PlannerExplodedError, match="planner exploded deterministically"):
        run_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=planner,
            planner_input=[ChatMessage(role="user", content="use tool")],
            allowed_tool_ids=("alpha.tool",),
            max_iterations=3,
        )


def test_bounded_tool_loop_budget_exceeded_propagates() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _OutA),
        _HandlerA(),
    )
    llm = _TwoRoundLLM()
    state = _runtime_state(llm)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = ToolPlanningService(llm=llm, tools=registry)

    with patch(
        "intergrax.runtime.nexus.tools.tool_loop.enforce_tool_call_budget",
        side_effect=BudgetExceededError("Budget exceeded: max_tool_calls (2 > 1)"),
    ):
        with pytest.raises(BudgetExceededError, match="max_tool_calls"):
            run_bounded_tool_loop(
                state=state,
                invoker=invoker,
                tool_planner=planner,
                planner_input=[ChatMessage(role="user", content="use tool")],
                allowed_tool_ids=("alpha.tool",),
                max_iterations=2,
            )


class _LongOutputTwoRoundLLM(FakeLLMAdapter):
    def __init__(self) -> None:
        super().__init__(fixed_text="")
        self._round = 0
        self.second_round_saw_marker = False

    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(self, messages, tools_schema, **kwargs):  # type: ignore[no-untyped-def]
        _ = tools_schema, kwargs
        self._round += 1
        if self._round == 1:
            return LLMAdapterResponse(
                content="",
                tool_calls=(
                    LLMToolCall.from_openai_shape(
                        call_id="tc-long",
                        name="long.tool",
                        arguments={},
                    ),
                ),
            )
        tool_messages = [msg for msg in messages if msg.role == "tool"]
        assert len(tool_messages) == 1
        tool_body = tool_messages[0].content or ""
        self.second_round_saw_marker = _BEYOND_PREVIEW_MARKER in tool_body
        if not self.second_round_saw_marker:
            return LLMAdapterResponse(content="missing tail marker", tool_calls=())
        return LLMAdapterResponse(content=f"confirmed:{_BEYOND_PREVIEW_MARKER}", tool_calls=())


class _FailThenRecoverLLM(FakeLLMAdapter):
    def __init__(self) -> None:
        super().__init__(fixed_text="")
        self._round = 0
        self.second_round_saw_marker = False

    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(self, messages, tools_schema, **kwargs):  # type: ignore[no-untyped-def]
        _ = tools_schema, kwargs
        self._round += 1
        if self._round == 1:
            return LLMAdapterResponse(
                content="",
                tool_calls=(
                    LLMToolCall.from_openai_shape(
                        call_id="tc-fail",
                        name="fail.tool",
                        arguments={},
                    ),
                ),
            )
        tool_messages = [msg for msg in messages if msg.role == "tool"]
        assert len(tool_messages) == 1
        tool_body = tool_messages[0].content or ""
        self.second_round_saw_marker = _BEYOND_PREVIEW_MARKER in tool_body
        if not self.second_round_saw_marker:
            return LLMAdapterResponse(content="missing tail marker", tool_calls=())
        return LLMAdapterResponse(content=f"confirmed:{_BEYOND_PREVIEW_MARKER}", tool_calls=())


def test_model_facing_tool_result_preserves_output_beyond_trace_preview() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("long.tool", _LongIn, _LongOut),
        _LongHandler(),
    )
    llm = _LongOutputTwoRoundLLM()
    state = _runtime_state(llm)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = ToolPlanningService(llm=llm, tools=registry)

    result = run_bounded_tool_loop(
        state=state,
        invoker=invoker,
        tool_planner=planner,
        planner_input=[ChatMessage(role="user", content="use long tool")],
        allowed_tool_ids=("long.tool",),
        max_iterations=2,
    )

    trace = result.tool_traces[0]
    full_json = _LongOut(padding="x" * 500, decision_token=_BEYOND_PREVIEW_MARKER).model_dump_json()
    tool_messages = [msg for msg in result.appended_messages if msg.role == "tool"]

    assert trace.output_preview is not None
    assert len(trace.output_preview) <= _PREVIEW_BOUND
    assert _BEYOND_PREVIEW_MARKER not in trace.output_preview
    assert len(tool_messages) == 1
    assert tool_messages[0].content == full_json
    assert llm.second_round_saw_marker is True
    assert result.stop_reason == "planner_final_answer"


def test_failed_tool_keeps_bounded_trace_and_model_facing_error() -> None:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("fail.tool", _FailIn, _FailOut),
        _FailHandler(),
    )
    llm = _FailThenRecoverLLM()
    state = _runtime_state(llm)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    planner = ToolPlanningService(llm=llm, tools=registry)

    result = run_bounded_tool_loop(
        state=state,
        invoker=invoker,
        tool_planner=planner,
        planner_input=[ChatMessage(role="user", content="use failing tool")],
        allowed_tool_ids=("fail.tool",),
        max_iterations=2,
    )

    trace = result.tool_traces[0]
    tool_messages = [msg for msg in result.appended_messages if msg.role == "tool"]
    full_error = f"RuntimeError: {'x' * 400 + _BEYOND_PREVIEW_MARKER}"

    assert trace.success is False
    assert trace.output_preview is None
    assert trace.error_message is not None
    assert len(trace.error_message) <= _PREVIEW_BOUND
    assert _BEYOND_PREVIEW_MARKER not in trace.error_message
    assert len(tool_messages) == 1
    assert tool_messages[0].content == full_error
    assert _BEYOND_PREVIEW_MARKER in tool_messages[0].content
    assert llm.second_round_saw_marker is True
    assert result.stop_reason == "planner_final_answer"

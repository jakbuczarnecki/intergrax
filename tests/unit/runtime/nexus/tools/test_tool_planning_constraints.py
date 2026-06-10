# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-4 / TOOL-ENG-11: planner constraints and context scope."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.config_types import ToolsContextScope
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.runtime_steps.tools_step import ToolsStep
from intergrax.runtime.nexus.tools.tool_planning_service import (
    ToolPlanningService,
    _build_openai_tools_schema,
)
from intergrax.runtime.nexus.tools.tool_planner_input import resolve_tool_planner_input
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler
from intergrax.tools.execution_models import ToolExecutionRequest
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager, tools_agent_make_contract

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _InA(BaseModel):
    value: int = 1


class _Out(BaseModel):
    result: int = 0


class _HandlerA(ToolHandler[_InA, _Out]):
    def execute(self, request: ToolExecutionRequest[_InA]) -> _Out:
        return _Out(result=request.input.value)


class _InB(BaseModel):
    query: str = ""


class _HandlerB(ToolHandler[_InB, _Out]):
    def execute(self, request: ToolExecutionRequest[_InB]) -> _Out:
        return _Out(result=len(request.input.query))


def _registry_with_two_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("alpha.tool", _InA, _Out),
        _HandlerA(),
    )
    registry.register(
        tools_agent_make_contract("beta.tool", _InB, _Out),
        _HandlerB(),
    )
    return registry


def test_build_openai_tools_schema_respects_allowed_tool_ids() -> None:
    registry = _registry_with_two_tools()
    full = _build_openai_tools_schema(registry)
    filtered = _build_openai_tools_schema(registry, allowed_tool_ids=["alpha.tool"])

    assert len(full) == 2
    assert len(filtered) == 1
    assert filtered[0]["function"]["name"] == "alpha.tool"


def test_tool_planning_service_native_path_filters_disallowed_tool_calls() -> None:
    registry = _registry_with_two_tools()
    llm = FakeLLMAdapter()

    class _ToolCall:
        def __init__(self, name: str, arguments_json: str) -> None:
            self.name = name
            self.arguments_json = arguments_json

    class _Result:
        tool_calls = [
            _ToolCall("beta.tool", '{"query":"x"}'),
            _ToolCall("alpha.tool", '{"value":2}'),
        ]

    llm.generate_with_tools = MagicMock(return_value=_Result())  # type: ignore[method-assign]
    llm.supports_tools = MagicMock(return_value=True)  # type: ignore[method-assign]

    service = ToolPlanningService(llm=llm, tools=registry)
    decision = service.plan_tools(
        "run tools",
        allowed_tool_ids=["alpha.tool"],
        run_id="run-plan",
    )

    assert decision.tool_plan is not None
    assert len(decision.tool_plan.calls) == 1
    assert decision.tool_plan.calls[0].tool_id == "alpha.tool"


def _runtime_state(*, scope: ToolsContextScope) -> RuntimeState:
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        enable_rag=False,
        enable_websearch=False,
        tools_context_scope=scope,
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
            message="latest question",
        ),
        run_id="run-scope",
        base_history=[ChatMessage(role="user", content="older")],
        messages_for_llm=[
            ChatMessage(role="system", content="system ctx"),
            ChatMessage(role="user", content="older"),
            ChatMessage(role="user", content="latest question"),
        ],
    )


def test_resolve_tool_planner_input_current_message_only() -> None:
    state = _runtime_state(scope=ToolsContextScope.CURRENT_MESSAGE_ONLY)
    assert resolve_tool_planner_input(state) == "latest question"


def test_resolve_tool_planner_input_conversation() -> None:
    state = _runtime_state(scope=ToolsContextScope.CONVERSATION)
    result = resolve_tool_planner_input(state)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[-1].content == "latest question"


def test_resolve_tool_planner_input_full() -> None:
    state = _runtime_state(scope=ToolsContextScope.FULL)
    result = resolve_tool_planner_input(state)
    assert result == state.messages_for_llm


@pytest.mark.asyncio
async def test_tools_step_passes_allowed_tool_ids_to_planner() -> None:
    state = _runtime_state(scope=ToolsContextScope.CURRENT_MESSAGE_ONLY)
    state.tool_planner_allowed_tool_ids = ("alpha.tool",)

    planner = MagicMock()
    planner.plan_tools = MagicMock(return_value=MagicMock(tool_plan=None))

    state.context.config.tool_invoker = MagicMock()
    state.context.config.tool_planner = planner
    state.context.config.tools_mode = "auto"

    await ToolsStep().run(state)

    planner.plan_tools.assert_called_once()
    kwargs = planner.plan_tools.call_args.kwargs
    assert kwargs["allowed_tool_ids"] == ("alpha.tool",)
    assert kwargs["input_data"] == "latest question"

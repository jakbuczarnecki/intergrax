# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-6 — bounded multi-iteration tool loop integration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.config_types import ToolsContextScope
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.runtime_steps.tool_loop_step import run_bounded_tool_loop
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.nexus.tools.tool_planning_service import ToolPlanningService
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.tool_executor import ToolHandler
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager, tools_agent_make_contract

pytestmark = [pytest.mark.integration, pytest.mark.gate]


class _InA(BaseModel):
    value: int = 1


class _OutA(BaseModel):
    result: int = 0


class _HandlerA(ToolHandler[_InA, _OutA]):
    def execute(self, request: ToolExecutionRequest[_InA]) -> _OutA:
        return _OutA(result=request.input.value)


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
            message="use tool",
        ),
        run_id="run-loop",
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

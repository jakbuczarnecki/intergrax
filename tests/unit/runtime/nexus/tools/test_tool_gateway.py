# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic import BaseModel

from intergrax.contracts.tool_request import ToolRequest, ToolResponseStatus
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.nexus.tools.tool_gateway import (
    NEXUS_CAPABILITY_PLAN,
    NEXUS_RAG,
    RuntimeToolGateway,
)
from intergrax.runtime.nexus.tools.tool_runtime import ToolRuntime, ToolRuntimeResult
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager, tools_agent_make_contract


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_tool_gateway_capability_plan_invokes_runtime():
    state = MagicMock()
    state.run_id = "run_1"
    state.used_rag = False
    state.used_websearch = False
    state.used_tools = False
    state.tool_traces = []

    with patch.object(
        ToolRuntime,
        "invoke",
        new_callable=AsyncMock,
        return_value=ToolRuntimeResult(
            used_rag=True,
            used_websearch=False,
            used_tools=False,
            tool_trace_count=0,
        ),
    ) as invoke_mock:
        gateway = RuntimeToolGateway.for_state(state, trace_step="TestGateway")
        response = await gateway.invoke(
            ToolRequest(
                tool_name=NEXUS_CAPABILITY_PLAN,
                agent_id="legal",
                step_id="s1",
                input={"use_rag": True, "use_websearch": False, "use_tools": False},
            )
        )

    assert response.status == ToolResponseStatus.SUCCESS
    assert response.output is not None
    assert response.output["used_rag"] is True
    invoke_mock.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_tool_gateway_denies_unknown_tool_when_not_allowed():
    state = MagicMock()
    state.run_id = "run_2"

    gateway = RuntimeToolGateway.for_state(state, allowed_tools=["allowed_only"])
    response = await gateway.invoke(
        ToolRequest(
            tool_name="custom.tool",
            agent_id="legal",
            step_id="s1",
            input={},
        )
    )

    assert response.status == ToolResponseStatus.DENIED


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_tool_gateway_single_rag_capability():
    state = MagicMock()
    state.run_id = "run_3"
    state.used_rag = True
    state.used_websearch = False
    state.used_tools = False
    state.tool_traces = []

    with patch.object(
        ToolRuntime,
        "invoke",
        new_callable=AsyncMock,
        return_value=ToolRuntimeResult(True, False, False, 0),
    ) as invoke_mock:
        gateway = RuntimeToolGateway.for_state(state)
        response = await gateway.invoke(
            ToolRequest(tool_name=NEXUS_RAG, agent_id="a1", step_id="s1")
        )

    assert response.status == ToolResponseStatus.SUCCESS
    plan = invoke_mock.await_args.kwargs["plan"]
    assert plan.tool_ids == (RAG_RETRIEVE_TOOL_ID,)
    assert plan.use_tools is False


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_tool_gateway_capability_plan_prefers_tool_ids_without_legacy_flags():
    state = MagicMock()
    state.run_id = "run_4"
    state.used_rag = False
    state.used_websearch = False
    state.used_tools = False
    state.tool_traces = []

    with patch.object(
        ToolRuntime,
        "invoke",
        new_callable=AsyncMock,
        return_value=ToolRuntimeResult(True, True, False, 0, tool_ids=(RAG_RETRIEVE_TOOL_ID,)),
    ) as invoke_mock:
        gateway = RuntimeToolGateway.for_state(state)
        response = await gateway.invoke(
            ToolRequest(
                tool_name=NEXUS_CAPABILITY_PLAN,
                agent_id="legal",
                step_id="s1",
                input={
                    "tool_ids": [RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID],
                    "use_tools": False,
                },
            )
        )

    assert response.status == ToolResponseStatus.SUCCESS
    plan = invoke_mock.await_args.kwargs["plan"]
    assert RAG_RETRIEVE_TOOL_ID in plan.tool_ids
    assert WEBSEARCH_QUERY_TOOL_ID in plan.tool_ids


class _GatewayIn(BaseModel):
    query: str = "default"


class _GatewayOut(BaseModel):
    hits: int = 0


class _GatewayHandler(ToolHandler[_GatewayIn, _GatewayOut]):
    def execute(self, request: ToolExecutionRequest[_GatewayIn]) -> _GatewayOut:
        return _GatewayOut(hits=len(request.input.query))


GATEWAY_CATALOG_TOOL = "jira.get_issue"


def _state_with_catalog_invoker() -> RuntimeState:
    registry = ToolRegistry()
    contract = tools_agent_make_contract(GATEWAY_CATALOG_TOOL, _GatewayIn, _GatewayOut)
    registry.register(contract, _GatewayHandler())
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
    )
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        enable_rag=False,
        enable_websearch=False,
        tool_invoker=invoker,
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
            message="issue",
        ),
        run_id="run-gw-catalog",
        tool_traces=[],
    )


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_tool_gateway_invokes_registered_catalog_tool():
    state = _state_with_catalog_invoker()
    gateway = RuntimeToolGateway.for_state(state, allowed_tools=[GATEWAY_CATALOG_TOOL])
    response = await gateway.invoke(
        ToolRequest(
            tool_name=GATEWAY_CATALOG_TOOL,
            agent_id="agent-1",
            step_id="s1",
            input={"query": "PROJ-1"},
        )
    )

    assert response.status == ToolResponseStatus.SUCCESS
    assert response.output is not None
    assert response.output["hits"] == 6

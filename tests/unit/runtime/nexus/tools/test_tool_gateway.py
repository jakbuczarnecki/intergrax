# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intergrax.contracts.tool_request import ToolRequest, ToolResponseStatus
from intergrax.runtime.nexus.tools.tool_gateway import (
    NEXUS_CAPABILITY_PLAN,
    NEXUS_RAG,
    RuntimeToolGateway,
)
from intergrax.runtime.nexus.tools.tool_runtime import ToolRuntime, ToolRuntimeResult


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
    assert plan.use_rag is True
    assert plan.use_websearch is False
    assert plan.use_tools is False

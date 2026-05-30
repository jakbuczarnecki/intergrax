# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from legal.domain.legal_tool_plan import LegalToolPlan
from legal.runtime.legal_tool_runtime_bridge import run_legal_tool_runtime_bridge
from intergrax.contracts.tool_request import ToolResponse, ToolResponseStatus
from intergrax.runtime.nexus.tools.tool_gateway import NEXUS_CAPABILITY_PLAN
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_legal_bridge_uses_tool_gateway_not_direct_runtime():
    state = MagicMock()
    state.request.metadata = {"allowed_tools": ["rag"]}
    state.trace_event = MagicMock()
    plan = LegalToolPlan(
        intent="rag",
        confidence=0.9,
        use_rag=True,
        use_websearch=False,
        use_tools=False,
    )

    with patch(
        "legal.runtime.legal_tool_runtime_bridge.RuntimeToolGateway.for_state",
    ) as factory:
        gateway = MagicMock()
        gateway.invoke = AsyncMock(
            return_value=ToolResponse(
                request_id="tool_test",
                status=ToolResponseStatus.SUCCESS,
                output={"used_rag": True},
            )
        )
        factory.return_value = gateway

        await run_legal_tool_runtime_bridge(state=state, plan=plan)

    factory.assert_called_once()
    gateway.invoke.assert_awaited_once()
    request = gateway.invoke.await_args.args[0]
    assert request.tool_name == NEXUS_CAPABILITY_PLAN
    assert request.input["use_rag"] is True
    assert RAG_RETRIEVE_TOOL_ID in request.input["tool_ids"]

# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-1: per-tool_id catalog dispatch in ToolRuntime."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.catalog_dispatch import (
    catalog_tool_ids,
    invoke_catalog_tool_ids,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.nexus.tools.tool_runtime import ToolInvocationPlan, ToolRuntime
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager, tools_agent_make_contract

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    result: int


class _Handler(ToolHandler[_In, _Out]):
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        return _Out(result=request.input.value + 10)


CATALOG_TOOL_ID = "jira.search_tasks"


def _register_catalog_tool(registry: ToolRegistry) -> None:
    contract = tools_agent_make_contract(CATALOG_TOOL_ID, _In, _Out)
    registry.register(contract, _Handler())


def _runtime_state_with_invoker(registry: ToolRegistry) -> RuntimeState:
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
            message="find issues",
        ),
        run_id="run-catalog-1",
        tool_traces=[],
    )


def test_catalog_tool_ids_excludes_pipeline_shims() -> None:
    ids = catalog_tool_ids([RAG_RETRIEVE_TOOL_ID, CATALOG_TOOL_ID])
    assert ids == (CATALOG_TOOL_ID,)


@pytest.mark.asyncio
async def test_invoke_catalog_tool_ids_without_use_tools_flag() -> None:
    registry = ToolRegistry()
    _register_catalog_tool(registry)
    state = _runtime_state_with_invoker(registry)

    count = invoke_catalog_tool_ids(
        state=state,
        tool_ids=[CATALOG_TOOL_ID],
        tool_inputs={CATALOG_TOOL_ID: {"value": 3}},
    )

    assert count == 1
    assert state.used_tools is True
    assert len(state.tool_traces) == 1
    assert state.tool_traces[0].tool_name == CATALOG_TOOL_ID
    assert state.tool_traces[0].success is True


@pytest.mark.asyncio
async def test_tool_runtime_invoke_dispatches_catalog_ids_without_use_tools() -> None:
    registry = ToolRegistry()
    _register_catalog_tool(registry)
    state = _runtime_state_with_invoker(registry)

    plan = ToolInvocationPlan.from_tool_ids([CATALOG_TOOL_ID])

    with (
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.run_rag_context",
            new_callable=AsyncMock,
        ),
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.run_websearch_context",
            new_callable=AsyncMock,
        ),
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.run_tools_context",
            new_callable=AsyncMock,
        ) as tools_step_mock,
    ):
        result = await ToolRuntime.invoke(state=state, plan=plan)

    tools_step_mock.assert_not_awaited()
    assert result.used_tools is True
    assert CATALOG_TOOL_ID in result.tool_ids
    assert len(state.tool_traces) == 1

# © Artur Czarnecki. All rights reserved.

"""P0-SAFETY-1C — ToolRuntime dynamic scope enforcement for legacy capabilities."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.nexus.tools.tool_runtime import ToolInvocationPlan, ToolRuntime
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler
from intergrax.tools.providers.rag.contracts import RagRetrieveInput, RagRetrieveOutput
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID
from testing_support.builder import (
    FakeLLMAdapter,
    build_in_memory_session_manager,
    canonical_execution_identity_scope,
    canonical_run_id_for_tests,
    tools_agent_make_contract,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _DenyToolPolicy:
    def __init__(self, denied_tool: str) -> None:
        self._denied_tool = denied_tool

    def is_allowed(self, *, agent_id: str, tool_id: str) -> bool:
        return tool_id != self._denied_tool


class _AgentScopedDenyPolicy:
    def __init__(self, *, denied_agent: str, denied_tool: str) -> None:
        self._denied_agent = denied_agent
        self._denied_tool = denied_tool

    def is_allowed(self, *, agent_id: str, tool_id: str) -> bool:
        if tool_id != self._denied_tool:
            return True
        return agent_id != self._denied_agent


def _runtime_request(*, agent_id: str, run_id: str) -> RuntimeRequest:
    return RuntimeRequest(
        agent_id=agent_id,
        user_id="user-1",
        session_id="session-1",
        tenant_id="tenant-1",
        message="find context",
        task_id="task_00000000000000000000000000000001",
        run_id=run_id,
    )


def _runtime_state(
    *,
    agent_id: str,
    enable_rag: bool = False,
    enable_websearch: bool = False,
    tool_scope_policy: object | None = None,
    tool_invoker: RuntimeToolInvoker | None = None,
) -> RuntimeState:
    run_id = canonical_run_id_for_tests("p0-safety-1c")
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        enable_rag=enable_rag,
        enable_websearch=enable_websearch,
        tool_scope_policy=tool_scope_policy,
        tool_invoker=tool_invoker,
    )
    ctx = RuntimeContext(
        config=config,
        session_manager=build_in_memory_session_manager(),
        prompt_registry=MagicMock(),
        context_builder=MagicMock(),
        rag_prompt_builder=MagicMock(),
        websearch_executor=MagicMock(),
        websearch_prompt_builder=MagicMock(),
    )
    state = RuntimeState(
        context=ctx,
        request=_runtime_request(agent_id=agent_id, run_id=run_id),
        run_id=run_id,
        session=ChatSession(id="session-1", tenant_id="tenant-1", user_id="user-1"),
        tool_traces=[],
    )
    return state


@pytest.mark.asyncio
async def test_legacy_rag_denied_by_dynamic_scope_policy() -> None:
    state = _runtime_state(
        agent_id="agent-1",
        enable_rag=True,
        tool_scope_policy=_DenyToolPolicy(RAG_RETRIEVE_TOOL_ID),
    )
    context_builder = state.context.context_builder
    assert context_builder is not None
    context_builder.build_context = AsyncMock()

    plan = ToolInvocationPlan.from_tool_ids([RAG_RETRIEVE_TOOL_ID])
    with canonical_execution_identity_scope(state.run_id):
        result = await ToolRuntime.invoke(state=state, plan=plan)

    context_builder.build_context.assert_not_awaited()
    assert result.used_rag is False
    assert state.used_rag is False


@pytest.mark.asyncio
async def test_legacy_websearch_denied_by_dynamic_scope_policy() -> None:
    state = _runtime_state(
        agent_id="agent-1",
        enable_websearch=True,
        tool_scope_policy=_DenyToolPolicy(WEBSEARCH_QUERY_TOOL_ID),
    )
    websearch_executor = state.context.websearch_executor
    assert websearch_executor is not None
    websearch_executor.search_async = AsyncMock()

    plan = ToolInvocationPlan.from_tool_ids([WEBSEARCH_QUERY_TOOL_ID])
    with canonical_execution_identity_scope(state.run_id):
        result = await ToolRuntime.invoke(state=state, plan=plan)

    websearch_executor.search_async.assert_not_awaited()
    assert result.used_websearch is False
    assert state.used_websearch is False


class _RagHandler(ToolHandler[RagRetrieveInput, RagRetrieveOutput]):
    def execute(self, request: ToolExecutionRequest[RagRetrieveInput]) -> RagRetrieveOutput:
        return RagRetrieveOutput(used=True, context_text="ctx")


@pytest.mark.asyncio
async def test_catalog_rag_uses_request_agent_id_for_scope_policy() -> None:
    registry = ToolRegistry()
    contract = tools_agent_make_contract(RAG_RETRIEVE_TOOL_ID, RagRetrieveInput, RagRetrieveOutput)
    registry.register(contract, _RagHandler())
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
        scope_policy=_AgentScopedDenyPolicy(
            denied_agent="customer-agent",
            denied_tool=RAG_RETRIEVE_TOOL_ID,
        ),
    )
    state = _runtime_state(
        agent_id="customer-agent",
        enable_rag=True,
        tool_scope_policy=_AgentScopedDenyPolicy(
            denied_agent="customer-agent",
            denied_tool=RAG_RETRIEVE_TOOL_ID,
        ),
        tool_invoker=invoker,
    )

    with patch.object(
        invoker,
        "invoke",
        wraps=invoker.invoke,
    ) as invoke_mock:
        plan = ToolInvocationPlan.from_tool_ids([RAG_RETRIEVE_TOOL_ID])
        with canonical_execution_identity_scope(state.run_id):
            result = await ToolRuntime.invoke(state=state, plan=plan)

    invoke_mock.assert_not_called()
    assert result.used_rag is False
    assert state.used_rag is False


class _WebIn(BaseModel):
    query: str = "q"
    limit: int = 3


class _WebOut(BaseModel):
    used: bool = True
    context_text: str = "ctx"


class _WebHandler(ToolHandler[_WebIn, _WebOut]):
    def execute(self, request: ToolExecutionRequest[_WebIn]) -> _WebOut:
        return _WebOut()


@pytest.mark.asyncio
async def test_catalog_websearch_uses_request_agent_id_for_scope_policy() -> None:
    registry = ToolRegistry()
    contract = tools_agent_make_contract(WEBSEARCH_QUERY_TOOL_ID, _WebIn, _WebOut)
    registry.register(contract, _WebHandler())
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
        scope_policy=_AgentScopedDenyPolicy(
            denied_agent="customer-agent",
            denied_tool=WEBSEARCH_QUERY_TOOL_ID,
        ),
    )
    state = _runtime_state(
        agent_id="customer-agent",
        enable_websearch=True,
        tool_scope_policy=_AgentScopedDenyPolicy(
            denied_agent="customer-agent",
            denied_tool=WEBSEARCH_QUERY_TOOL_ID,
        ),
        tool_invoker=invoker,
    )

    with patch.object(
        invoker,
        "invoke",
        wraps=invoker.invoke,
    ) as invoke_mock:
        plan = ToolInvocationPlan.from_tool_ids([WEBSEARCH_QUERY_TOOL_ID])
        with canonical_execution_identity_scope(state.run_id):
            result = await ToolRuntime.invoke(state=state, plan=plan)

    invoke_mock.assert_not_called()
    assert result.used_websearch is False
    assert state.used_websearch is False


@pytest.mark.asyncio
async def test_catalog_rag_passes_request_agent_id_to_invoker() -> None:
    registry = ToolRegistry()
    contract = tools_agent_make_contract(RAG_RETRIEVE_TOOL_ID, RagRetrieveInput, RagRetrieveOutput)
    registry.register(contract, _RagHandler())
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
        scope_policy=_AgentScopedDenyPolicy(
            denied_agent="customer-agent",
            denied_tool=RAG_RETRIEVE_TOOL_ID,
        ),
    )
    state = _runtime_state(
        agent_id="nexus",
        enable_rag=True,
        tool_scope_policy=_AgentScopedDenyPolicy(
            denied_agent="customer-agent",
            denied_tool=RAG_RETRIEVE_TOOL_ID,
        ),
        tool_invoker=invoker,
    )
    captured_agent_ids: list[str] = []
    original_invoke = invoker.invoke

    def _capture_invoke(**kwargs: object) -> ToolExecutionResult[BaseModel]:
        captured_agent_ids.append(str(kwargs["agent_id"]))
        return original_invoke(**kwargs)

    with (
        patch.object(invoker, "invoke", side_effect=_capture_invoke),
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.record_rag_invocation_and_enforce",
        ),
    ):
        plan = ToolInvocationPlan.from_tool_ids([RAG_RETRIEVE_TOOL_ID])
        with canonical_execution_identity_scope(state.run_id):
            result = await ToolRuntime.invoke(state=state, plan=plan)

    assert captured_agent_ids == ["nexus"]
    assert result.used_rag is True

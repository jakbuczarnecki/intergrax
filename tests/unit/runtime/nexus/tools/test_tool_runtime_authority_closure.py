# © Artur Czarnecki. All rights reserved.

"""P0-SAFETY-1D — empty-scope and direct StaticToolScopePolicy closure."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.config_types import ToolSelectionMode
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.nexus.tools.tool_invocation_pattern import ToolInvocationResult
from intergrax.runtime.nexus.tools.tool_runtime import ToolInvocationPlan, ToolRuntime
from intergrax.runtime.nexus.tools.tool_selection import (
    ToolSelectionContext,
    resolve_planner_allowed_tool_ids,
)
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID
from testing_support.builder import (
    FakeLLMAdapter,
    build_in_memory_session_manager,
    canonical_execution_identity_scope,
    canonical_run_id_for_tests,
    tools_agent_make_contract,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _SelectiveScopePolicy:
    def __init__(self, *, allowed: set[str]) -> None:
        self._allowed = frozenset(allowed)

    def is_allowed(self, *, agent_id: str, tool_id: str) -> bool:
        return tool_id in self._allowed


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    result: int = 0


class _EchoHandler(ToolHandler[_In, _Out]):
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        _ = request
        return _Out(result=1)


def _runtime_request(*, agent_id: str, run_id: str) -> RuntimeRequest:
    return RuntimeRequest(
        agent_id=agent_id,
        user_id="user-1",
        session_id="session-1",
        tenant_id="tenant-1",
        message="use tools",
        task_id="task_00000000000000000000000000000001",
        run_id=run_id,
    )


def _runtime_state(
    *,
    agent_id: str = "agent-1",
    enable_rag: bool = False,
    enable_websearch: bool = False,
    tool_scope_policy: object | None = None,
    tool_invoker: RuntimeToolInvoker | None = None,
) -> RuntimeState:
    run_id = canonical_run_id_for_tests("p0-safety-1d")
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        enable_rag=enable_rag,
        enable_websearch=enable_websearch,
        tool_scope_policy=tool_scope_policy,
        tool_invoker=tool_invoker,
        tool_planner=MagicMock(),
        tools_mode="auto",
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
    return RuntimeState(
        context=ctx,
        request=_runtime_request(agent_id=agent_id, run_id=run_id),
        run_id=run_id,
        session=ChatSession(id="session-1", tenant_id="tenant-1", user_id="user-1"),
        tool_traces=[],
    )


def _registry_with(*tool_ids: str) -> ToolRegistry:
    registry = ToolRegistry()
    for tool_id in tool_ids:
        registry.register(
            tools_agent_make_contract(tool_id, _In, _Out),
            _EchoHandler(),
        )
    return registry


@pytest.mark.asyncio
async def test_direct_static_scope_policy_denies_legacy_rag() -> None:
    state = _runtime_state(
        enable_rag=True,
        tool_scope_policy=StaticToolScopePolicy(allowed_tools=set()),
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
async def test_direct_static_scope_policy_denies_legacy_websearch() -> None:
    state = _runtime_state(
        enable_websearch=True,
        tool_scope_policy=StaticToolScopePolicy(allowed_tools=set()),
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


@pytest.mark.asyncio
async def test_empty_scope_after_dynamic_filtering_blocks_planner_widening() -> None:
    registry = _registry_with("tool.a", "tool.b")
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
    )
    state = _runtime_state(
        tool_scope_policy=_SelectiveScopePolicy(allowed={"tool.b"}),
        tool_invoker=invoker,
    )
    captured_allowed: list[tuple[str, ...] | None] = []

    async def _capture_loop(**kwargs: object) -> ToolInvocationResult:
        allowed = kwargs.get("allowed_tool_ids")
        captured_allowed.append(tuple(allowed) if allowed is not None else ())
        return ToolInvocationResult()

    plan = ToolInvocationPlan(tool_ids=("tool.a",), use_tools=True)
    with (
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.run_bounded_tool_loop_async",
            side_effect=_capture_loop,
        ),
        patch.object(invoker, "invoke", wraps=invoker.invoke) as invoke_mock,
        canonical_execution_identity_scope(state.run_id),
    ):
        result = await ToolRuntime.invoke(
            state=state,
            plan=plan,
            allowed_tools=["tool.a"],
        )

    assert captured_allowed == [()]
    invoke_mock.assert_not_called()
    assert result.used_tools is False
    assert state.tool_traces == []


def test_resolve_planner_allowed_tool_ids_empty_tuple_is_authoritative() -> None:
    registry = _registry_with("tool.a", "tool.b")
    ctx = ToolSelectionContext(
        registry=registry,
        query="tool search",
        plan_allowed_tool_ids=(),
        top_k=1,
    )
    ids = resolve_planner_allowed_tool_ids(ToolSelectionMode.RETRIEVAL_TOP_K, ctx)
    assert ids == ()


def test_resolve_planner_allowed_tool_ids_none_keeps_strategy_candidates() -> None:
    registry = _registry_with("tool.a", "tool.b")
    ctx = ToolSelectionContext(
        registry=registry,
        query="tool search",
        plan_allowed_tool_ids=None,
        top_k=1,
    )
    ids = resolve_planner_allowed_tool_ids(ToolSelectionMode.RETRIEVAL_TOP_K, ctx)
    assert ids is not None
    assert len(ids) == 1

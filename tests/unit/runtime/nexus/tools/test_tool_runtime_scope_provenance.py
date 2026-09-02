# © Artur Czarnecki. All rights reserved.

"""P0-SAFETY-1E — authoritative tool scope provenance closure."""

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
from intergrax.runtime.nexus.tools.tool_invocation_pattern import ToolInvocationResult
from intergrax.runtime.nexus.tools.tool_runtime import ToolInvocationPlan, ToolRuntime
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler
from testing_support.builder import (
    FakeLLMAdapter,
    build_in_memory_session_manager,
    canonical_execution_identity_scope,
    canonical_run_id_for_tests,
    tools_agent_make_contract,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


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
    policy_bundle: RuntimePolicyBundle | None = None,
    tool_invoker: RuntimeToolInvoker | None = None,
) -> RuntimeState:
    run_id = canonical_run_id_for_tests("p0-safety-1e")
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        policy_bundle=policy_bundle,
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
async def test_explicit_empty_allowed_tools_blocks_planner_widening() -> None:
    registry = _registry_with("tool.a", "tool.b")
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
    )
    state = _runtime_state(tool_invoker=invoker)
    captured_allowed: list[tuple[str, ...] | None] = []

    async def _capture_loop(**kwargs: object) -> ToolInvocationResult:
        allowed = kwargs.get("allowed_tool_ids")
        captured_allowed.append(tuple(allowed) if allowed is not None else None)
        return ToolInvocationResult()

    plan = ToolInvocationPlan(tool_ids=(), use_tools=True)
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
            allowed_tools=[],
        )

    assert captured_allowed == [()]
    invoke_mock.assert_not_called()
    assert result.used_tools is False
    assert state.tool_traces == []


@pytest.mark.asyncio
async def test_upstream_empty_bundle_scope_blocks_planner_widening() -> None:
    registry = _registry_with("tool.a", "tool.b")
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
    )
    bundle = RuntimePolicyBundle(
        tool_access=StaticToolScopePolicy(allowed_tools=set()),
    )
    state = _runtime_state(tool_invoker=invoker, policy_bundle=bundle)
    captured_allowed: list[tuple[str, ...] | None] = []

    async def _capture_loop(**kwargs: object) -> ToolInvocationResult:
        allowed = kwargs.get("allowed_tool_ids")
        captured_allowed.append(tuple(allowed) if allowed is not None else None)
        return ToolInvocationResult()

    plan = ToolInvocationPlan(tool_ids=(), use_tools=True)
    with (
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.run_bounded_tool_loop_async",
            side_effect=_capture_loop,
        ),
        patch.object(invoker, "invoke", wraps=invoker.invoke) as invoke_mock,
        canonical_execution_identity_scope(state.run_id),
    ):
        result = await ToolRuntime.invoke(state=state, plan=plan)

    assert captured_allowed == [()]
    invoke_mock.assert_not_called()
    assert result.used_tools is False
    assert state.tool_traces == []


@pytest.mark.asyncio
async def test_unconstrained_empty_plan_keeps_planner_unrestricted() -> None:
    registry = _registry_with("tool.a", "tool.b")
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
    )
    state = _runtime_state(tool_invoker=invoker)
    captured_allowed: list[tuple[str, ...] | None] = []

    async def _capture_loop(**kwargs: object) -> ToolInvocationResult:
        allowed = kwargs.get("allowed_tool_ids")
        captured_allowed.append(tuple(allowed) if allowed is not None else None)
        return ToolInvocationResult()

    plan = ToolInvocationPlan(tool_ids=(), use_tools=True)
    with (
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.run_bounded_tool_loop_async",
            side_effect=_capture_loop,
        ),
        canonical_execution_identity_scope(state.run_id),
    ):
        await ToolRuntime.invoke(state=state, plan=plan)

    assert captured_allowed == [None]

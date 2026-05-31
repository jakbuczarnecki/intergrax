# © Artur Czarnecki. All rights reserved.

import pytest

from echo.echo_agent import EchoAgent
from intergrax.contracts.tool_request import ToolRequest, ToolResponseStatus
from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.nexus.execution.graph_builder import plan_to_execution_graph
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.planning.task_planner import NexusPlan, PlanStep
from intergrax.runtime.nexus.tools.tool_gateway import RuntimeToolGateway
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.builder import build_runtime_state_for_tests


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_tool_gateway_before_tool_call_hook_blocks() -> None:
    pipeline = MiddlewarePipeline()

    async def block_tool(_ctx: HookContext) -> HookResult:
        return HookResult(action=HookAction.BLOCK, reason="tool denied")

    pipeline.hooks.register(HookPoint.BEFORE_TOOL_CALL, block_tool)

    state = build_runtime_state_for_tests(run_id="tool-hook-run")
    gateway = RuntimeToolGateway.for_state(state, middleware=pipeline)
    response = await gateway.invoke(
        ToolRequest(
            request_id="r1",
            tool_name="rag",
            agent_id="echo",
            step_id="s1",
            input={},
        ),
    )
    assert response.status == ToolResponseStatus.DENIED
    assert "tool denied" in (response.error or "")


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_graph_executor_agent_selection_hook_blocks() -> None:
    pipeline = MiddlewarePipeline()

    async def block_selection(_ctx: HookContext) -> HookResult:
        return HookResult(action=HookAction.BLOCK, reason="selection denied")

    pipeline.hooks.register(HookPoint.BEFORE_AGENT_SELECTION, block_selection)

    registry = AgentRegistry()
    registry.register(EchoAgent())
    executor = GraphExecutor(registry, middleware=pipeline)

    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="hi",
        context=TaskContext(capability="echo.basic"),
    )
    plan = NexusPlan(
        plan_id="p1",
        task_id=task.task_id,
        classification="single_agent",
        steps=[
            PlanStep(step_id="s1", agent_id="echo", capability="echo.basic"),
        ],
    )
    graph = plan_to_execution_graph(plan)

    executions, _retries, _graph, success = await executor.execute(graph, task)
    assert success is False
    assert executions == []


@pytest.mark.unit
@pytest.mark.gate
def test_legal_factory_exposes_interaction_intake_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEGAL_INCLUDE_MCP", "false")
    from legal_application.host.factory import create_legal_backend_app

    app = create_legal_backend_app()
    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/v1/interactions/intake" in paths

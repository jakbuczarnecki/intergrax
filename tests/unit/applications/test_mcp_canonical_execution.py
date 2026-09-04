# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.contracts.execution_identity import (
    mint_run_id,
    mint_task_id,
    validate_execution_id,
)
from intergrax.runtime.execution.facade import Execution as ExecutionFacade
from intergrax.applications._shared.host_task_execution_wiring import build_host_task_execution
from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver
from intergrax.runtime.execution.strategy_router import StrategyExecutionRouter
from intergrax.runtime.execution.task_adapter import TaskExecutionInput
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import TaskResult, TaskState
from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
)
from governed_contractor_application.host.execution_wiring import (
    build_governed_contractor_host_task_execution,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.mcp.server import build_governed_contractor_mcp_server
from research_application.mcp.server import build_research_mcp_server

pytestmark = [pytest.mark.unit]


def _build_governed_contractor_host_execution(nexus_loop: NexusLoop):
    env = build_governed_contractor_environment_profile(GovernedContractorBackendSettings.from_env())
    return build_governed_contractor_host_task_execution(nexus_loop, env)


def _build_research_host_execution(nexus_loop: NexusLoop):
    return build_host_task_execution(
        nexus_loop,
        orchestration_triggers=frozenset(),
        pipeline_capability_suffix=".pipeline",
    )


@pytest.mark.asyncio
async def test_shared_mcp_run_agent_uses_canonical_execution_facade() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    host_execution = _build_governed_contractor_host_execution(nexus_loop)
    mcp = build_nexus_mcp_server(
        name="Shared MCP Test",
        host_execution=host_execution,
        registry=registry,
        default_capability="external_contractor.adapt",
    )
    facade_calls = 0
    original_execute = ExecutionFacade.execute

    async def _spy_execute(self, request, *, options):
        nonlocal facade_calls
        facade_calls += 1
        return await original_execute(self, request, options=options)

    with patch.object(ExecutionFacade, "execute", _spy_execute):
        with patch(
            "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
            new_callable=AsyncMock,
            return_value=TaskResult(
                task_id=mint_task_id(),
                run_id=mint_run_id(),
                state=TaskState.COMPLETED,
                answer="ok",
            ),
        ):
            from fastmcp import Client

            async with Client(mcp) as client:
                result = await client.call_tool(
                    "run_agent",
                    {"message": "facade proof", "capability": "external_contractor.adapt"},
                )

    assert facade_calls == 1
    assert result.data["state"] == TaskState.COMPLETED.value


@pytest.mark.asyncio
async def test_governed_contractor_mcp_adapt_does_not_root_call_nexus_handle_task() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    nexus_loop.handle_task = AsyncMock()
    host_execution = _build_governed_contractor_host_execution(nexus_loop)
    mcp = build_governed_contractor_mcp_server(
        host_execution=host_execution,
        registry=registry,
        route_prefix="/v1/governed_contractor",
    )

    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        new_callable=AsyncMock,
        return_value=TaskResult(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="ok",
        ),
    ):
        from fastmcp import Client

        async with Client(mcp) as client:
            await client.call_tool(
                "run_agent",
                {"message": "adapt proof", "capability": "external_contractor.adapt"},
            )

    nexus_loop.handle_task.assert_not_called()


@pytest.mark.asyncio
async def test_governed_contractor_mcp_adapt_reaches_strategy_router_with_agentic_capability() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    host_execution = _build_governed_contractor_host_execution(nexus_loop)
    mcp = build_governed_contractor_mcp_server(
        host_execution=host_execution,
        registry=registry,
        route_prefix="/v1/governed_contractor",
    )
    captured: dict[str, object] = {}

    async def _capture_execute(
        self: StrategyExecutionRouter[TaskExecutionInput, TaskResult, TaskResult],
        request: ExecutionRequest[TaskExecutionInput, TaskResult],
    ) -> TaskResult:
        captured["strategy"] = StrategyResolver().resolve(request)
        captured["capabilities"] = request.capabilities
        return TaskResult(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="routed",
        )

    with patch.object(StrategyExecutionRouter, "execute", _capture_execute):
        from fastmcp import Client

        async with Client(mcp) as client:
            await client.call_tool(
                "run_agent",
                {"message": "strategy proof", "capability": "external_contractor.adapt"},
            )

    assert captured["strategy"] is ExecutionStrategy.AGENTIC
    assert captured["capabilities"] == frozenset({ExecutionCapability.AGENT})


@pytest.mark.asyncio
async def test_research_mcp_pipeline_reaches_orchestration_strategy() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    host_execution = _build_research_host_execution(nexus_loop)
    mcp = build_research_mcp_server(
        host_execution=host_execution,
        registry=registry,
        route_prefix="/v1/research",
    )
    captured: dict[str, object] = {}

    async def _capture_execute(
        self: StrategyExecutionRouter[TaskExecutionInput, TaskResult, TaskResult],
        request: ExecutionRequest[TaskExecutionInput, TaskResult],
    ) -> TaskResult:
        captured["strategy"] = StrategyResolver().resolve(request)
        captured["capabilities"] = request.capabilities
        return TaskResult(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="pipeline",
        )

    with patch.object(StrategyExecutionRouter, "execute", _capture_execute):
        from fastmcp import Client

        async with Client(mcp) as client:
            await client.call_tool(
                "run_research_pipeline",
                {"message": "orchestration proof"},
            )

    assert captured["strategy"] is ExecutionStrategy.ORCHESTRATION
    assert captured["capabilities"] == frozenset({ExecutionCapability.ORCHESTRATION})


@pytest.mark.asyncio
async def test_mcp_root_execution_id_is_platform_owned() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    host_execution = _build_governed_contractor_host_execution(nexus_loop)
    mcp = build_governed_contractor_mcp_server(
        host_execution=host_execution,
        registry=registry,
        route_prefix="/v1/governed_contractor",
    )
    observed: dict[str, object] = {}

    async def _capture_execute(
        self: StrategyExecutionRouter[TaskExecutionInput, TaskResult, TaskResult],
        request: ExecutionRequest[TaskExecutionInput, TaskResult],
    ) -> TaskResult:
        from intergrax.contracts.execution_identity import require_active_execution_id

        observed["execution_id"] = require_active_execution_id()
        return TaskResult(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="identity",
        )

    with patch.object(StrategyExecutionRouter, "execute", _capture_execute):
        from fastmcp import Client

        async with Client(mcp) as client:
            await client.call_tool(
                "run_agent",
                {
                    "message": "identity proof",
                    "capability": "external_contractor.adapt",
                },
            )

    active_execution_id = observed["execution_id"]
    assert isinstance(active_execution_id, str)
    validate_execution_id(active_execution_id)


@pytest.mark.asyncio
async def test_mcp_request_produces_single_root_execution_invocation() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    host_execution = _build_governed_contractor_host_execution(nexus_loop)
    mcp = build_governed_contractor_mcp_server(
        host_execution=host_execution,
        registry=registry,
        route_prefix="/v1/governed_contractor",
    )
    router_calls = 0
    facade_calls = 0
    original_facade_execute = ExecutionFacade.execute

    async def _count_facade_execute(self, request, *, options):
        nonlocal facade_calls
        facade_calls += 1
        return await original_facade_execute(self, request, options=options)

    async def _count_router_execute(
        self: StrategyExecutionRouter[TaskExecutionInput, TaskResult, TaskResult],
        request: ExecutionRequest[TaskExecutionInput, TaskResult],
    ) -> TaskResult:
        nonlocal router_calls
        router_calls += 1
        return TaskResult(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="one",
        )

    with patch.object(ExecutionFacade, "execute", _count_facade_execute):
        with patch.object(StrategyExecutionRouter, "execute", _count_router_execute):
            from fastmcp import Client

            async with Client(mcp) as client:
                await client.call_tool(
                    "run_agent",
                    {"message": "single root", "capability": "external_contractor.adapt"},
                )

    assert facade_calls == 1
    assert router_calls == 1

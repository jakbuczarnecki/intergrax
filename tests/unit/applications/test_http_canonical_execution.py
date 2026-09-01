# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable
from unittest.mock import AsyncMock, patch

import pytest

from intergrax.applications._shared.host_task_execution_wiring import build_environment_host_task_execution
from intergrax.contracts.execution_identity import (
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    validate_execution_id,
)
from intergrax.runtime.execution.facade import Execution as ExecutionFacade
from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver
from intergrax.runtime.execution.strategy_router import StrategyExecutionRouter
from intergrax.runtime.execution.task_adapter import TaskExecutionInput
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from dispute_sim_application.host.environment_profile import build_dispute_sim_environment_profile
from dispute_sim_application.host.settings import DisputeSimBackendSettings
from dispute_sim_application.serving.fastapi_router import DisputeSimRunService
from dispute_sim_application.serving.schemas import DisputeSimRunRequestV1
from research_application.host.wiring import build_research_environment_profile
from research_application.host.settings import ResearchBackendSettings
from research_application.serving.fastapi_router import ResearchRunService
from research_application.serving.schemas import ResearchRunRequestV1

pytestmark = [pytest.mark.unit]


@dataclass(frozen=True)
class _HttpExecutionCase:
    label: str
    build_service: Callable[[NexusLoop], object]
    invoke: Callable[[object], Awaitable[object]]
    capability: str
    expected_strategy: ExecutionStrategy
    expected_capabilities: frozenset[ExecutionCapability]


def _build_dispute_service(nexus_loop: NexusLoop) -> DisputeSimRunService:
    env = build_dispute_sim_environment_profile(DisputeSimBackendSettings.from_env())
    host_execution = build_environment_host_task_execution(nexus_loop, env)
    return DisputeSimRunService.from_host_execution(
        host_execution,
        default_agent_id="echo",
    )


async def _invoke_dispute(service: object) -> object:
    assert isinstance(service, DisputeSimRunService)
    return await service.run_task(
        DisputeSimRunRequestV1(message="http proof", capability="dispute.intake")
    )


def _build_research_service(nexus_loop: NexusLoop) -> ResearchRunService:
    env = build_research_environment_profile(ResearchBackendSettings.from_env())
    host_execution = build_environment_host_task_execution(nexus_loop, env)
    return ResearchRunService.from_host_execution(host_execution)


async def _invoke_research(service: object) -> object:
    assert isinstance(service, ResearchRunService)
    return await service.run_pipeline(
        ResearchRunRequestV1(message="http proof"),
        authenticated_principal=None,
    )


_HTTP_EXECUTION_CASES = (
    _HttpExecutionCase(
        label="dispute_sim",
        build_service=_build_dispute_service,
        invoke=_invoke_dispute,
        capability="dispute.intake",
        expected_strategy=ExecutionStrategy.AGENTIC,
        expected_capabilities=frozenset({ExecutionCapability.AGENT}),
    ),
    _HttpExecutionCase(
        label="research",
        build_service=_build_research_service,
        invoke=_invoke_research,
        capability="research.pipeline",
        expected_strategy=ExecutionStrategy.ORCHESTRATION,
        expected_capabilities=frozenset({ExecutionCapability.ORCHESTRATION}),
    ),
)


@pytest.mark.parametrize("case", _HTTP_EXECUTION_CASES, ids=lambda case: case.label)
@pytest.mark.asyncio
async def test_http_root_uses_canonical_execution_facade(case: _HttpExecutionCase) -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    service = case.build_service(nexus_loop)
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
            with patch(
                "intergrax.runtime.execution.host_task.TaskBoundOrchestrationDelegate.execute",
                new_callable=AsyncMock,
                return_value=TaskResult(
                    task_id=mint_task_id(),
                    run_id=mint_run_id(),
                    state=TaskState.COMPLETED,
                    answer="ok",
                ),
            ):
                await case.invoke(service)

    assert facade_calls == 1


@pytest.mark.parametrize("case", _HTTP_EXECUTION_CASES, ids=lambda case: case.label)
@pytest.mark.asyncio
async def test_http_request_produces_single_root_execution_invocation(case: _HttpExecutionCase) -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    service = case.build_service(nexus_loop)
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
            await case.invoke(service)

    assert facade_calls == 1
    assert router_calls == 1


@pytest.mark.parametrize("case", _HTTP_EXECUTION_CASES, ids=lambda case: case.label)
@pytest.mark.asyncio
async def test_http_reaches_strategy_router_with_expected_capability(case: _HttpExecutionCase) -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    service = case.build_service(nexus_loop)
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
        await case.invoke(service)

    assert captured["strategy"] is case.expected_strategy
    assert captured["capabilities"] == case.expected_capabilities


@pytest.mark.asyncio
async def test_http_agentic_request_does_not_root_call_nexus_handle_task() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    nexus_loop.handle_task = AsyncMock()  # type: ignore[method-assign]
    service = _build_dispute_service(nexus_loop)

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
        await _invoke_dispute(service)

    nexus_loop.handle_task.assert_not_called()


@pytest.mark.asyncio
async def test_http_root_execution_id_is_platform_owned() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    service = _build_dispute_service(nexus_loop)
    caller_execution_id = mint_execution_id()
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
        await service.run_task(
            DisputeSimRunRequestV1(
                message="identity proof",
                capability="dispute.intake",
                metadata={"execution_id": caller_execution_id},
            )
        )

    active_execution_id = observed["execution_id"]
    assert isinstance(active_execution_id, str)
    validate_execution_id(active_execution_id)
    assert active_execution_id != caller_execution_id


@pytest.mark.asyncio
async def test_http_and_mcp_dispute_capability_resolve_to_same_strategy() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    env = build_dispute_sim_environment_profile(DisputeSimBackendSettings.from_env())
    build_environment_host_task_execution(nexus_loop, env)
    capability = "dispute.intake"
    http_task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="parity",
        context=TaskContext(capability=capability),
    )
    mcp_task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="parity",
        context=TaskContext(capability=capability),
    )
    from intergrax.runtime.execution.host_task import resolve_task_execution_capabilities
    from intergrax.runtime.execution.task_adapter import execution_request_from_task
    from intergrax.runtime.nexus.orchestration_capabilities import orchestration_capabilities_from_triggers

    graph_spec = env.graph_spec
    orchestration_triggers = orchestration_capabilities_from_triggers(
        graph_spec.trigger_capabilities if graph_spec is not None else None,
    )
    pipeline_suffix = (
        graph_spec.pipeline_capability_suffix if graph_spec is not None else ".pipeline"
    )
    http_caps = resolve_task_execution_capabilities(
        http_task,
        orchestration_triggers=orchestration_triggers,
        pipeline_capability_suffix=pipeline_suffix,
    )
    mcp_caps = resolve_task_execution_capabilities(
        mcp_task,
        orchestration_triggers=orchestration_triggers,
        pipeline_capability_suffix=pipeline_suffix,
    )
    resolver = StrategyResolver()
    http_request = execution_request_from_task(
        http_task,
        capabilities=http_caps,
        output_type=TaskResult,
    )
    mcp_request = execution_request_from_task(
        mcp_task,
        capabilities=mcp_caps,
        output_type=TaskResult,
    )
    assert resolver.resolve(http_request) is resolver.resolve(mcp_request)

# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from pathlib import Path

import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    validate_execution_id,
)
from intergrax.runtime.execution.facade import Execution as ExecutionFacade
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver
from intergrax.runtime.execution.strategy_router import StrategyExecutionRouter
from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.task_adapter import TaskExecutionInput
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.execution_wiring import build_lkw_host_task_execution
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.host.task_executor import LocalWorkspaceTaskExecutor

pytestmark = [pytest.mark.unit]


def _settings() -> LocalWorkspaceBackendSettings:
    return LocalWorkspaceBackendSettings.from_env()


def _build_executor(nexus_loop: NexusLoop) -> LocalWorkspaceTaskExecutor:
    env = build_local_workspace_environment_profile(_settings())
    host_execution = build_lkw_host_task_execution(nexus_loop, env)
    from local_workspace_application.host.lifecycle import LocalWorkspaceHostLifecycle

    lifecycle = LocalWorkspaceHostLifecycle()
    lifecycle.set_executor_available(True)
    lifecycle.transition_to_ready()
    return LocalWorkspaceTaskExecutor(host_execution, task_enricher=None, readiness=lifecycle)


@pytest.mark.asyncio
async def test_lkw_application_root_uses_canonical_execution_facade() -> None:
    settings = _settings()
    env = build_local_workspace_environment_profile(settings)
    host_execution = build_lkw_host_task_execution(NexusLoop(AgentRegistry()), env)
    assert isinstance(host_execution.execution, ExecutionFacade)
    factory_source = (
        Path(__file__).resolve().parents[2] / "host" / "factory.py"
    ).read_text(encoding="utf-8")
    assert "build_lkw_host_task_execution" in factory_source
    assert "UnifiedTaskRunner" not in factory_source.split("lkw_task_executor", 1)[0]


@pytest.mark.asyncio
async def test_lkw_search_does_not_root_call_nexus_handle_task() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    nexus_loop.handle_task = AsyncMock()  # type: ignore[method-assign]
    executor = _build_executor(nexus_loop)
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="search proof",
        context=TaskContext(capability="local.workspace.search"),
    )

    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        new_callable=AsyncMock,
        return_value=TaskResult(
            task_id=task.task_id,
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="ok",
        ),
    ):
        await executor.execute(task)

    nexus_loop.handle_task.assert_not_called()


@pytest.mark.asyncio
async def test_lkw_search_reaches_strategy_router_with_agentic_capability() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    executor = _build_executor(nexus_loop)
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="search proof",
        context=TaskContext(capability="local.workspace.search"),
    )
    captured: dict[str, object] = {}

    original_execute = StrategyExecutionRouter.execute

    async def _capture_execute(
        self: StrategyExecutionRouter[TaskExecutionInput, TaskResult, TaskResult],
        request: ExecutionRequest[TaskExecutionInput, TaskResult],
    ) -> TaskResult:
        captured["strategy"] = StrategyResolver().resolve(request)
        captured["capabilities"] = request.capabilities
        return TaskResult(
            task_id=task.task_id,
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="routed",
        )

    with patch.object(StrategyExecutionRouter, "execute", _capture_execute):
        await executor.execute(task)

    assert captured["strategy"] is ExecutionStrategy.AGENTIC
    assert captured["capabilities"] == frozenset({ExecutionCapability.AGENT})


@pytest.mark.asyncio
async def test_lkw_pipeline_reaches_orchestration_strategy() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    executor = _build_executor(nexus_loop)
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="pipeline proof",
        context=TaskContext(capability="local.workspace.pipeline"),
    )
    captured: dict[str, object] = {}

    async def _capture_execute(
        self: StrategyExecutionRouter[TaskExecutionInput, TaskResult, TaskResult],
        request: ExecutionRequest[TaskExecutionInput, TaskResult],
    ) -> TaskResult:
        captured["strategy"] = StrategyResolver().resolve(request)
        return TaskResult(
            task_id=task.task_id,
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="pipeline",
        )

    with patch.object(StrategyExecutionRouter, "execute", _capture_execute):
        await executor.execute(task)

    assert captured["strategy"] is ExecutionStrategy.ORCHESTRATION


@pytest.mark.asyncio
async def test_lkw_root_execution_id_is_platform_owned() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    executor = _build_executor(nexus_loop)
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="identity proof",
        context=TaskContext(capability="local.workspace.search"),
    )
    caller_execution_id = mint_execution_id()
    task = task.model_copy(
        update={
            "metadata": {
                **task.metadata,
                "execution_id": caller_execution_id,
            }
        }
    )
    observed: dict[str, object] = {}

    async def _capture_execute(
        self: StrategyExecutionRouter[TaskExecutionInput, TaskResult, TaskResult],
        request: ExecutionRequest[TaskExecutionInput, TaskResult],
    ) -> TaskResult:
        from intergrax.contracts.execution_identity import require_active_execution_id

        observed["execution_id"] = require_active_execution_id()
        return TaskResult(
            task_id=task.task_id,
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="identity",
        )

    with patch.object(StrategyExecutionRouter, "execute", _capture_execute):
        await executor.execute(task)

    active_execution_id = observed["execution_id"]
    assert isinstance(active_execution_id, str)
    validate_execution_id(active_execution_id)
    assert active_execution_id != caller_execution_id


@pytest.mark.asyncio
async def test_lkw_request_produces_single_root_execution_invocation() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    executor = _build_executor(nexus_loop)
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="single root",
        context=TaskContext(capability="local.workspace.search"),
    )
    calls = 0

    async def _count_execute(
        self: StrategyExecutionRouter[TaskExecutionInput, TaskResult, TaskResult],
        request: ExecutionRequest[TaskExecutionInput, TaskResult],
    ) -> TaskResult:
        nonlocal calls
        calls += 1
        return TaskResult(
            task_id=task.task_id,
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="one",
        )

    with patch.object(StrategyExecutionRouter, "execute", _count_execute):
        await executor.execute(task)

    assert calls == 1

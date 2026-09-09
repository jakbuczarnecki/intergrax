# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from intergrax.contracts.agent_execution_result import (
    AgentExecutionResult,
    AgentExecutionStatus,
)
from intergrax.contracts.execution_identity import (
    mint_task_id,
    require_active_execution_identity,
)
from intergrax.contracts.execution_lineage import build_execution_lineage_attempt_scope
from intergrax.runtime.execution.nexus_host_execution import build_host_task_execution
from intergrax.runtime.execution.lineage.persistence import (
    InMemoryExecutionLineagePersistence,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState


@pytest.mark.asyncio
async def test_build_host_task_execution_wires_lineage_persistence() -> None:
    persistence = InMemoryExecutionLineagePersistence()
    nexus_loop = NexusLoop(AgentRegistry(), execution_lineage_persistence=persistence)
    host_execution = build_host_task_execution(
        nexus_loop, orchestration_triggers=frozenset()
    )
    assert host_execution._execution_lineage_persistence is persistence


@pytest.mark.asyncio
async def test_host_task_root_admission_exists_before_delegate() -> None:
    persistence = InMemoryExecutionLineagePersistence()
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry, execution_lineage_persistence=persistence)
    host_execution = build_host_task_execution(
        nexus_loop, orchestration_triggers=frozenset()
    )
    task_id = mint_task_id()
    task = Task(
        task_id=task_id,
        tenant_id="tenant-a",
        user_id="user-a",
        message="lineage proof",
        context=TaskContext(capability="agent.demo"),
        agent_id="demo-agent",
    )
    checked_before_delegate = False

    async def _run_with_result(runtime_request: object) -> AgentExecutionResult:
        nonlocal checked_before_delegate
        run_id, attempt_id = require_active_execution_identity()
        scope = build_execution_lineage_attempt_scope(
            tenant_id=task.tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
        )
        page = persistence.list_admissions_for_attempt(scope, limit=10)
        assert len(page.admissions) == 1
        checked_before_delegate = True
        return AgentExecutionResult(
            agent_id="demo-agent",
            run_id=run_id,
            status=AgentExecutionStatus.COMPLETED,
            summary="ok",
        )

    nexus_loop._engine.run_with_result = AsyncMock(side_effect=_run_with_result)  # noqa: SLF001

    result = await host_execution.execute(task)
    assert checked_before_delegate is True
    assert result.task_id == task_id
    assert result.state is TaskState.COMPLETED

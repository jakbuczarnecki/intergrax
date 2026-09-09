# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from intergrax.contracts.agent_execution_result import (
    AgentExecutionResult,
    AgentExecutionStatus,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageSegmentLifecycle,
    build_execution_lineage_attempt_scope,
)
from intergrax.runtime.execution.active_execution_resume import (
    peek_active_execution_resume_plan,
)
from intergrax.runtime.execution.nexus_host_execution import build_host_task_execution
from intergrax.runtime.execution.lineage.persistence import (
    InMemoryExecutionLineagePersistence,
)
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    minimal_runtime_checkpoint,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState


def _checkpoint(
    *,
    task_id: str,
    run_id: RunId,
    attempt_id: AttemptId,
    root_execution_id: ExecutionId,
) -> TaskCheckpoint:
    return TaskCheckpoint(
        task_id=task_id,
        tenant_id="tenant-a",
        resume_token="rt-host-resume",
        task_state=TaskState.WAITING_FOR_HUMAN,
        runtime=minimal_runtime_checkpoint(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            root_execution_id=root_execution_id,
        ),
    )


@pytest.mark.asyncio
async def test_host_task_resume_uses_single_canonical_root_execution_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistence = InMemoryExecutionLineagePersistence()
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry, execution_lineage_persistence=persistence)
    host_execution = build_host_task_execution(
        nexus_loop, orchestration_triggers=frozenset()
    )
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    checkpoint_root = mint_execution_id()
    minted_roots: list[ExecutionId] = []
    original_mint = mint_execution_id

    def _track_mint() -> ExecutionId:
        execution_id = original_mint()
        minted_roots.append(execution_id)
        return execution_id

    monkeypatch.setattr(
        "intergrax.runtime.execution.identity_authority.mint_execution_id",
        _track_mint,
    )
    checkpoint = _checkpoint(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        root_execution_id=checkpoint_root,
    )
    scope = build_execution_lineage_attempt_scope(
        tenant_id="tenant-a",
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    persistence.open_attempt(scope)
    persistence.open_segment(scope, checkpoint_root)
    task = Task(
        task_id=task_id,
        tenant_id="tenant-a",
        user_id="user-a",
        message="resume lineage identity",
        context=TaskContext(capability="agent.demo"),
        agent_id="demo-agent",
    )
    captured: dict[str, ExecutionId] = {}

    async def _run_with_result(runtime_request: object) -> AgentExecutionResult:
        del runtime_request
        active_run_id, active_attempt_id = require_active_execution_identity()
        active_execution_id = require_active_execution_id()
        resume_plan = peek_active_execution_resume_plan()
        assert resume_plan is not None
        active_snapshot = resume_plan.plan.active_snapshot
        resume_root = next(
            entry.execution_id
            for entry in active_snapshot.entries
            if entry.parent_execution_id is None
        )
        scope = build_execution_lineage_attempt_scope(
            tenant_id=task.tenant_id,
            task_id=task_id,
            run_id=active_run_id,
            attempt_id=active_attempt_id,
        )
        attempt_state = persistence.read_attempt_lineage_state(scope)
        assert attempt_state is not None
        page = persistence.list_admissions_for_attempt(scope, limit=10)
        root_admissions = [
            item for item in page.admissions if item.parent_execution_id is None
        ]
        assert len(root_admissions) == 1
        captured["resume_plan_root"] = resume_root
        captured["active_execution_id"] = active_execution_id
        captured["segment_root"] = attempt_state.active_segment_root_execution_id
        captured["root_admission"] = root_admissions[0].execution_id
        return AgentExecutionResult(
            agent_id="demo-agent",
            run_id=active_run_id,
            status=AgentExecutionStatus.COMPLETED,
            summary="ok",
        )

    nexus_loop._engine.run_with_result = AsyncMock(side_effect=_run_with_result)  # noqa: SLF001

    result = await host_execution.execute(
        task,
        resume_checkpoint=checkpoint,
        execution_id=None,
    )

    assert result.state is TaskState.COMPLETED
    assert len(minted_roots) == 1
    canonical_root = minted_roots[0]
    assert canonical_root != checkpoint_root
    assert captured["resume_plan_root"] == canonical_root
    assert captured["active_execution_id"] == canonical_root
    assert captured["segment_root"] == canonical_root
    assert captured["root_admission"] == canonical_root


@pytest.mark.asyncio
async def test_host_task_resume_same_attempt_persists_two_segment_roots() -> None:
    persistence = InMemoryExecutionLineagePersistence()
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry, execution_lineage_persistence=persistence)
    host_execution = build_host_task_execution(
        nexus_loop, orchestration_triggers=frozenset()
    )
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    task = Task(
        task_id=task_id,
        tenant_id="tenant-a",
        user_id="user-a",
        message="segment continuity",
        context=TaskContext(capability="agent.demo"),
        agent_id="demo-agent",
    )

    first_root: ExecutionId | None = None

    async def _run_with_result(runtime_request: object) -> AgentExecutionResult:
        nonlocal first_root
        del runtime_request
        active_run_id, active_attempt_id = require_active_execution_identity()
        first_root = require_active_execution_id()
        return AgentExecutionResult(
            agent_id="demo-agent",
            run_id=active_run_id,
            status=AgentExecutionStatus.COMPLETED,
            summary="ok",
        )

    nexus_loop._engine.run_with_result = AsyncMock(side_effect=_run_with_result)  # noqa: SLF001

    await host_execution.execute(task, run_id=run_id, attempt_id=attempt_id)
    assert first_root is not None
    scope = build_execution_lineage_attempt_scope(
        tenant_id=task.tenant_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    first_segment = persistence.open_segment(scope, first_root)
    assert first_segment.lifecycle is ExecutionLineageSegmentLifecycle.SEGMENT_OPEN

    checkpoint = _checkpoint(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        root_execution_id=first_root,
    )
    resume_task = Task(
        task_id=task_id,
        tenant_id="tenant-a",
        user_id="user-a",
        message="segment continuity resume",
        context=TaskContext(capability="agent.demo"),
        agent_id="demo-agent",
    )
    resume_root: ExecutionId | None = None

    async def _run_resume(runtime_request: object) -> AgentExecutionResult:
        nonlocal resume_root
        del runtime_request
        active_run_id, active_attempt_id = require_active_execution_identity()
        resume_root = require_active_execution_id()
        return AgentExecutionResult(
            agent_id="demo-agent",
            run_id=active_run_id,
            status=AgentExecutionStatus.COMPLETED,
            summary="ok",
        )

    nexus_loop._engine.run_with_result = AsyncMock(side_effect=_run_resume)  # noqa: SLF001

    await host_execution.execute(
        resume_task,
        resume_checkpoint=checkpoint,
        execution_id=None,
    )

    assert resume_root is not None
    assert resume_root != first_root
    second_segment = persistence.open_segment(scope, resume_root, first_root)
    assert second_segment.predecessor_root_execution_id == first_root
    first_segment_after = persistence.open_segment(scope, first_root)
    assert (
        first_segment_after.lifecycle
        is ExecutionLineageSegmentLifecycle.SEGMENT_UNCLEAN
    )
    attempt_state = persistence.read_attempt_lineage_state(scope)
    assert attempt_state is not None
    assert attempt_state.active_segment_root_execution_id == resume_root
    assert {first_root, resume_root} == {
        first_segment.root_execution_id,
        second_segment.root_execution_id,
    }

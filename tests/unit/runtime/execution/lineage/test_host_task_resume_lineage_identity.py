# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
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
from intergrax.runtime.execution.lineage.persistence import (
    InMemoryExecutionLineagePersistence,
)
from intergrax.runtime.execution.nexus_host_execution import build_host_task_execution
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    minimal_runtime_checkpoint,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent


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


class _LineageIdentityProbeAgent(UaepPipelineStubAgent):
    """Registered UAEP gate stub that observes active execution identity during delegate."""

    __slots__ = (
        "_capture_resume_details",
        "_captured",
        "_persistence",
        "_task_id",
        "_task_tenant_id",
    )

    def __init__(
        self,
        *,
        captured: dict[str, ExecutionId],
        persistence: InMemoryExecutionLineagePersistence | None = None,
        task_id: str | None = None,
        task_tenant_id: str = "tenant-a",
        capture_resume_details: bool = False,
    ) -> None:
        super().__init__(
            agent_id="demo-agent",
            capability="agent.demo",
            prefix="demo",
        )
        self._captured = captured
        self._persistence = persistence
        self._task_id = task_id
        self._task_tenant_id = task_tenant_id
        self._capture_resume_details = capture_resume_details

    async def run_step(
        self,
        step: AgentStep,
        ctx: RuntimeExecutionContext,
    ) -> StepOutput:
        active_run_id, active_attempt_id = require_active_execution_identity()
        active_execution_id = require_active_execution_id()
        if self._capture_resume_details:
            resume_plan = peek_active_execution_resume_plan()
            assert resume_plan is not None
            active_snapshot = resume_plan.plan.active_snapshot
            resume_root = next(
                entry.execution_id
                for entry in active_snapshot.entries
                if entry.parent_execution_id is None
            )
            assert self._persistence is not None
            assert self._task_id is not None
            attempt_scope = build_execution_lineage_attempt_scope(
                tenant_id=self._task_tenant_id,
                task_id=self._task_id,
                run_id=active_run_id,
                attempt_id=active_attempt_id,
            )
            attempt_state = self._persistence.read_attempt_lineage_state(attempt_scope)
            assert attempt_state is not None
            page = self._persistence.list_admissions_for_attempt(
                attempt_scope, limit=10
            )
            root_admissions = [
                item for item in page.admissions if item.parent_execution_id is None
            ]
            assert len(root_admissions) == 1
            self._captured["resume_plan_root"] = resume_root
            self._captured["active_execution_id"] = active_execution_id
            self._captured["segment_root"] = (
                attempt_state.active_segment_root_execution_id
            )
            self._captured["root_admission"] = root_admissions[0].execution_id
        else:
            self._captured["active_execution_id"] = active_execution_id
        return await super().run_step(step, ctx)


def _build_host_execution(
    persistence: InMemoryExecutionLineagePersistence,
    *,
    captured: dict[str, ExecutionId],
    task_id: str | None = None,
    capture_resume_details: bool = False,
):
    registry = AgentRegistry()
    registry.register(
        _LineageIdentityProbeAgent(
            captured=captured,
            persistence=persistence,
            task_id=task_id,
            capture_resume_details=capture_resume_details,
        ),
    )
    nexus_loop = NexusLoop(registry, execution_lineage_persistence=persistence)
    return build_host_task_execution(nexus_loop, orchestration_triggers=frozenset())


@pytest.mark.asyncio
async def test_host_task_resume_uses_single_canonical_root_execution_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistence = InMemoryExecutionLineagePersistence()
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
    host_execution = _build_host_execution(
        persistence,
        captured=captured,
        task_id=task_id,
        capture_resume_details=True,
    )

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

    first_captured: dict[str, ExecutionId] = {}
    host_execution = _build_host_execution(persistence, captured=first_captured)

    await host_execution.execute(task, run_id=run_id, attempt_id=attempt_id)
    first_root = first_captured.get("active_execution_id")
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
    resume_captured: dict[str, ExecutionId] = {}
    resume_host_execution = _build_host_execution(
        persistence,
        captured=resume_captured,
    )

    await resume_host_execution.execute(
        resume_task,
        resume_checkpoint=checkpoint,
        execution_id=None,
    )

    resume_root = resume_captured.get("active_execution_id")
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

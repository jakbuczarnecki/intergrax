# © Artur Czarnecki. All rights reserved.

import pytest
from pydantic import ValidationError

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.runtime.long_running.checkpoint_builder import (
    apply_runtime_checkpoint_to_graph,
    build_runtime_checkpoint,
    should_resume_uaep_step,
    should_skip_graph_node,
    should_skip_uaep_step,
)
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    ExecutionCheckpointEntry,
    ExecutionCheckpointStatus,
    ExecutionPriorOutput,
    ExecutionTreeRecorder,
    ExecutionTreeSnapshot,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.runtime_checkpoint import (
    RuntimeCheckpoint,
    UaepStepCursor,
    UaepStepOutput,
)
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode, ExecutionNodeStatus
from intergrax.runtime.nexus.planning.task_planner import NexusPlan, PlanStep
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_contract import HumanApprovalResolution
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner


def _tree(
    *,
    task_id: str,
    run_id: str,
    attempt_id: str,
    root_execution_id: str,
) -> ExecutionTreeSnapshot:
    return ExecutionTreeSnapshot(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        entries=[
            ExecutionCheckpointEntry(
                execution_id=root_execution_id,
                parent_execution_id=None,
                status=ExecutionCheckpointStatus.RUNNING,
            )
        ],
    )


def _runtime(*, task_id: str, run_id: str, attempt_id: str, root_execution_id: str) -> RuntimeCheckpoint:
    return RuntimeCheckpoint(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_tree=_tree(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            root_execution_id=root_execution_id,
        ),
    )


def _approve_resolution(
    *,
    pause_id: str = "pause-1",
    human_request_id: str = "hr-1",
    task_id: str = "task-1",
) -> HumanApprovalResolution:
    return HumanApprovalResolution(
        task_id=task_id,
        pause_id=pause_id,
        human_request_id=human_request_id,
        verdict=HumanResponseVerdict.APPROVE,
        approver=local_development_approver_evidence(tenant_id="default"),
        resolved_at="2026-08-18T00:00:00+00:00",
    )


@pytest.mark.unit
@pytest.mark.gate
def test_runtime_checkpoint_accepts_valid_canonical_identity():
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    task_id = mint_task_id()
    ckpt = _runtime(task_id=task_id, run_id=run_id, attempt_id=attempt_id, root_execution_id=root)
    assert ckpt.run_id == run_id
    assert ckpt.attempt_id == attempt_id


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    "payload",
    [
        {"attempt_id": mint_attempt_id()},
        {"run_id": mint_run_id()},
        {},
    ],
)
def test_runtime_checkpoint_missing_identity_fails(payload):
    with pytest.raises(ValidationError):
        RuntimeCheckpoint.model_validate(payload)


@pytest.mark.unit
@pytest.mark.gate
def test_runtime_checkpoint_wrong_run_id_prefix_fails():
    with pytest.raises(ValidationError, match="RunId must start with 'run_'"):
        RuntimeCheckpoint(
            run_id="task_0123456789abcdef0123456789abcdef",
            attempt_id=mint_attempt_id(),
            execution_tree=ExecutionTreeSnapshot(
                task_id=mint_task_id(),
                run_id="task_0123456789abcdef0123456789abcdef",
                attempt_id=mint_attempt_id(),
                entries=[
                    ExecutionCheckpointEntry(
                        execution_id=mint_execution_id(),
                        parent_execution_id=None,
                        status=ExecutionCheckpointStatus.RUNNING,
                    )
                ],
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_should_skip_uaep_step_when_resumed_at_same_index():
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    task_id = mint_task_id()
    root = mint_execution_id()
    ckpt = RuntimeCheckpoint(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_tree=_tree(task_id=task_id, run_id=run_id, attempt_id=attempt_id, root_execution_id=root),
        uaep_step_index=0,
        uaep_step_id="review",
        last_step_output=UaepStepOutput(step_id="review", summary="pending"),
    )
    assert should_skip_uaep_step(
        step_index=0,
        step_id="review",
        checkpoint=ckpt,
        approval=_approve_resolution(),
    )


@pytest.mark.unit
@pytest.mark.gate
def test_should_resume_uaep_step_when_mid_step_cursor_present():
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    task_id = mint_task_id()
    root = mint_execution_id()
    ckpt = RuntimeCheckpoint(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_tree=_tree(task_id=task_id, run_id=run_id, attempt_id=attempt_id, root_execution_id=root),
        uaep_step_index=0,
        uaep_step_id="process",
        uaep_step_completed=False,
        uaep_step_cursor=UaepStepCursor(values={"phase1_done": True}),
        last_step_output=UaepStepOutput(step_id="process", summary="phase1 partial"),
    )
    assert should_resume_uaep_step(
        step_index=0,
        step_id="process",
        checkpoint=ckpt,
        approval=_approve_resolution(),
    )


@pytest.mark.unit
@pytest.mark.gate
def test_build_runtime_checkpoint_includes_plan_and_graph_snapshots():
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=root,
    )
    try:
        task = Task(tenant_id="t1", user_id="u1", agent_id="a1", task_id=task_id)
        plan = NexusPlan(
            task_id=task_id,
            classification="single_agent",
            steps=[PlanStep(step_id="step_1", agent_id="a1")],
        )
        graph = ExecutionGraph(
            graph_id="g_full",
            task_id=task_id,
            nodes=[
                ExecutionNode(node_id="n1", agent_id="a1", status=ExecutionNodeStatus.COMPLETED),
                ExecutionNode(node_id="n2", agent_id="a2"),
            ],
        )
        runtime = build_runtime_checkpoint(
            task,
            run_id=run_id,
            attempt_id=attempt_id,
            plan=plan,
            graph=graph,
        )
    finally:
        reset_active_execution_identity(token)
    assert isinstance(runtime, RuntimeCheckpoint)
    assert runtime.plan_snapshot is not None
    assert runtime.graph_snapshot is not None
    assert runtime.node_states["n1"] == ExecutionNodeStatus.COMPLETED.value


@pytest.mark.unit
@pytest.mark.gate
def test_should_skip_graph_node_when_completed_with_prior_output():
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    task_id = mint_task_id()
    root = mint_execution_id()
    child = mint_execution_id()
    ckpt = RuntimeCheckpoint(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_tree=ExecutionTreeSnapshot(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            entries=[
                ExecutionCheckpointEntry(
                    execution_id=root,
                    parent_execution_id=None,
                    status=ExecutionCheckpointStatus.RUNNING,
                ),
                ExecutionCheckpointEntry(
                    execution_id=child,
                    parent_execution_id=root,
                    status=ExecutionCheckpointStatus.COMPLETED,
                    graph_node_id="n1",
                    prior_output=ExecutionPriorOutput(
                        agent_id="a1",
                        summary="done",
                        status="completed",
                        graph_node_id="n1",
                    ),
                ),
            ],
        ),
        node_states={"n1": ExecutionNodeStatus.COMPLETED.value},
    )
    node = ExecutionNode(node_id="n1", agent_id="a1", status=ExecutionNodeStatus.COMPLETED)
    prior = {
        "n1": AgentExecutionResult(
            agent_id="a1",
            run_id=run_id,
            status=AgentExecutionStatus.COMPLETED,
            summary="done",
        )
    }
    assert should_skip_graph_node(node, checkpoint=ckpt, prior_outputs=prior)


@pytest.mark.unit
@pytest.mark.gate
def test_apply_runtime_checkpoint_to_graph_restores_completed_node():
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    child = mint_execution_id()
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=task_id,
        nodes=[
            ExecutionNode(node_id="n1", agent_id="a1"),
            ExecutionNode(node_id="n2", agent_id="a2", depends_on=["n1"]),
        ],
    )
    runtime = RuntimeCheckpoint(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_tree=ExecutionTreeSnapshot(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            entries=[
                ExecutionCheckpointEntry(
                    execution_id=root,
                    parent_execution_id=None,
                    status=ExecutionCheckpointStatus.RUNNING,
                ),
                ExecutionCheckpointEntry(
                    execution_id=child,
                    parent_execution_id=root,
                    status=ExecutionCheckpointStatus.COMPLETED,
                    graph_node_id="n1",
                    prior_output=ExecutionPriorOutput(
                        agent_id="a1",
                        summary="step one",
                        status="completed",
                        graph_node_id="n1",
                    ),
                ),
            ],
        ),
        node_states={
            "n1": ExecutionNodeStatus.COMPLETED.value,
            "n2": ExecutionNodeStatus.FAILED.value,
        },
    )
    prior: dict = {}
    apply_runtime_checkpoint_to_graph(graph, runtime, prior, run_id=run_id)

    assert graph.node_by_id("n1").status == ExecutionNodeStatus.COMPLETED
    assert graph.node_by_id("n1").execution_result is not None
    assert graph.node_by_id("n1").execution_result.summary == "step one"
    assert graph.node_by_id("n2").status == ExecutionNodeStatus.FAILED
    assert "n1" in prior


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_unified_task_runner_resume_preserves_checkpoint_identity(monkeypatch):
    from intergrax.runtime.nexus.nexus_loop import NexusLoop
    from intergrax.runtime.registry.agent_registry import AgentRegistry

    captured: dict[str, object] = {}
    loop = NexusLoop(AgentRegistry())
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    task_id = mint_task_id()
    root = mint_execution_id()

    async def _fake_handle_task(task: Task, *, run_id, attempt_id=None) -> TaskResult:
        captured["run_id"] = run_id
        captured["attempt_id"] = attempt_id
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "handle_task", _fake_handle_task)
    runner = UnifiedTaskRunner(loop)
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="resume", task_id=task_id)
    checkpoint = TaskCheckpoint(
        task_id=task.task_id,
        tenant_id="t1",
        resume_token="rt_test",
        task_state=TaskState.WAITING_FOR_HUMAN,
        runtime=_runtime(task_id=task_id, run_id=run_id, attempt_id=attempt_id, root_execution_id=root),
    )

    await runner.run_task(task, resume_checkpoint=checkpoint)

    assert captured["run_id"] == run_id
    assert captured["attempt_id"] == attempt_id


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_unified_task_runner_resume_rejects_explicit_run_id_mismatch(monkeypatch):
    from intergrax.runtime.nexus.nexus_loop import NexusLoop
    from intergrax.runtime.registry.agent_registry import AgentRegistry

    loop = NexusLoop(AgentRegistry())
    checkpoint_run_id = mint_run_id()
    checkpoint_attempt_id = mint_attempt_id()
    task_id = mint_task_id()
    root = mint_execution_id()
    runner = UnifiedTaskRunner(loop)
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="resume", task_id=task_id)
    checkpoint = TaskCheckpoint(
        task_id=task.task_id,
        tenant_id="t1",
        resume_token="rt_test",
        task_state=TaskState.WAITING_FOR_HUMAN,
        runtime=_runtime(
            task_id=task_id,
            run_id=checkpoint_run_id,
            attempt_id=checkpoint_attempt_id,
            root_execution_id=root,
        ),
    )

    with pytest.raises(ValueError, match="explicit run_id conflicts"):
        await runner.run_task(
            task,
            run_id=mint_run_id(),
            attempt_id=checkpoint_attempt_id,
            resume_checkpoint=checkpoint,
        )

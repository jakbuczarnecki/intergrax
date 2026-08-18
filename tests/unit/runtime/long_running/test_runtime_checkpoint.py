# © Artur Czarnecki. All rights reserved.

import pytest
from pydantic import ValidationError

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
from intergrax.runtime.long_running.checkpoint_builder import (
    apply_runtime_checkpoint_to_graph,
    build_runtime_checkpoint,
    should_resume_uaep_step,
    should_skip_graph_node,
    should_skip_uaep_step,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.runtime_checkpoint import (
    RUNTIME_CHECKPOINT_KEY,
    RuntimeCheckpoint,
    RuntimeCheckpointExecutionState,
    runtime_checkpoint_from_execution_structured,
    runtime_checkpoint_from_metadata,
)
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode, ExecutionNodeStatus
from intergrax.runtime.nexus.planning.task_planner import NexusPlan, PlanStep
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_contract import HumanApprovalResolution
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner


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
        resolved_at="2026-08-18T00:00:00+00:00",
    )


@pytest.mark.unit
@pytest.mark.gate
def test_runtime_checkpoint_accepts_valid_canonical_identity():
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    ckpt = RuntimeCheckpoint(run_id=run_id, attempt_id=attempt_id)
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
        )


@pytest.mark.unit
@pytest.mark.gate
def test_runtime_checkpoint_wrong_attempt_id_prefix_fails():
    with pytest.raises(ValidationError, match="AttemptId must start with 'attempt_'"):
        RuntimeCheckpoint(
            run_id=mint_run_id(),
            attempt_id="run_0123456789abcdef0123456789abcdef",
        )


@pytest.mark.unit
@pytest.mark.gate
def test_runtime_checkpoint_malformed_identity_fails():
    with pytest.raises(ValidationError, match="suffix must match"):
        RuntimeCheckpoint(run_id="run_tooshort", attempt_id=mint_attempt_id())


@pytest.mark.unit
@pytest.mark.gate
def test_runtime_checkpoint_serializes_flat_string_identity():
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    payload = RuntimeCheckpoint(run_id=run_id, attempt_id=attempt_id).model_dump(mode="json")
    assert payload["run_id"] == run_id
    assert payload["attempt_id"] == attempt_id
    assert isinstance(payload["run_id"], str)
    assert isinstance(payload["attempt_id"], str)


@pytest.mark.unit
@pytest.mark.gate
def test_runtime_checkpoint_from_execution_structured_legacy_boundary():
    structured = {
        RUNTIME_CHECKPOINT_KEY: {
            "uaep_step_index": 1,
            "uaep_step_id": "review",
            "last_step_output": {"step_id": "review", "summary": "done"},
        }
    }
    ckpt = runtime_checkpoint_from_execution_structured(structured)
    assert ckpt is not None
    assert isinstance(ckpt, RuntimeCheckpointExecutionState)
    assert ckpt.uaep_step_index == 1
    assert ckpt.uaep_step_id == "review"
    assert ckpt.run_id is None
    assert ckpt.attempt_id is None


@pytest.mark.unit
@pytest.mark.gate
def test_runtime_checkpoint_from_metadata_legacy_boundary():
    metadata = {
        RUNTIME_CHECKPOINT_KEY: {
            "uaep_step_index": 2,
            "run_id": mint_run_id(),
            "attempt_id": mint_attempt_id(),
        }
    }
    ckpt = runtime_checkpoint_from_metadata(metadata)
    assert ckpt is not None
    assert isinstance(ckpt, RuntimeCheckpointExecutionState)
    assert ckpt.uaep_step_index == 2


@pytest.mark.unit
@pytest.mark.gate
def test_should_skip_uaep_step_when_resumed_at_same_index():
    ckpt = RuntimeCheckpointExecutionState(
        uaep_step_index=0,
        uaep_step_id="review",
        last_step_output={"step_id": "review", "summary": "pending"},
    )
    assert should_skip_uaep_step(
        step_index=0,
        step_id="review",
        checkpoint=ckpt,
        approval=_approve_resolution(),
    )
    assert not should_skip_uaep_step(
        step_index=0,
        step_id="review",
        checkpoint=ckpt,
        approval=None,
    )
    assert not should_skip_uaep_step(
        step_index=1,
        step_id="review",
        checkpoint=ckpt,
        approval=_approve_resolution(),
    )


@pytest.mark.unit
@pytest.mark.gate
def test_should_resume_uaep_step_when_mid_step_cursor_present():
    ckpt = RuntimeCheckpointExecutionState(
        uaep_step_index=0,
        uaep_step_id="process",
        uaep_step_completed=False,
        uaep_step_cursor={"phase1_done": True},
        last_step_output={"step_id": "process", "summary": "phase1 partial"},
    )
    assert should_resume_uaep_step(
        step_index=0,
        step_id="process",
        checkpoint=ckpt,
        approval=_approve_resolution(),
    )
    assert not should_skip_uaep_step(
        step_index=0,
        step_id="process",
        checkpoint=ckpt,
        approval=_approve_resolution(),
    )
    completed = ckpt.model_copy(update={"uaep_step_completed": True, "uaep_step_cursor": None})
    assert should_skip_uaep_step(
        step_index=0,
        step_id="process",
        checkpoint=completed,
        approval=_approve_resolution(),
    )


@pytest.mark.unit
@pytest.mark.gate
def test_build_runtime_checkpoint_includes_plan_and_graph_snapshots():
    task_id = mint_task_id()
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
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        plan=plan,
        graph=graph,
    )
    assert isinstance(runtime, RuntimeCheckpoint)
    assert runtime.plan_snapshot is not None
    assert runtime.plan_snapshot["plan_id"] == plan.plan_id
    assert runtime.graph_snapshot is not None
    assert runtime.graph_snapshot["graph_id"] == "g_full"
    assert runtime.node_states["n1"] == ExecutionNodeStatus.COMPLETED.value


@pytest.mark.unit
@pytest.mark.gate
def test_build_runtime_checkpoint_requires_canonical_identity():
    task = Task(tenant_id="t1", user_id="u1", agent_id="a1")
    with pytest.raises(ValidationError):
        build_runtime_checkpoint(task, run_id="run_bad", attempt_id=mint_attempt_id())


@pytest.mark.unit
@pytest.mark.gate
def test_build_runtime_checkpoint_merges_execution_structured():
    task = Task(tenant_id="t1", user_id="u1", agent_id="hitl")
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution = AgentExecutionResult(
        agent_id="hitl",
        run_id=run_id,
        status=AgentExecutionStatus.NEEDS_INPUT,
        summary="awaiting approval",
        structured_data={
            RUNTIME_CHECKPOINT_KEY: {
                "uaep_step_index": 0,
                "uaep_step_id": "review",
                "last_step_output": {"step_id": "review", "summary": "pending review"},
            }
        },
    )
    runtime = build_runtime_checkpoint(
        task,
        run_id=run_id,
        attempt_id=attempt_id,
        last_execution=execution,
    )
    assert runtime.run_id == run_id
    assert runtime.attempt_id == attempt_id
    assert runtime.uaep_step_index == 0
    assert runtime.uaep_step_id == "review"
    assert runtime.last_step_output is not None


@pytest.mark.unit
@pytest.mark.gate
def test_should_skip_graph_node_when_completed_with_prior_output():
    ckpt = RuntimeCheckpointExecutionState(
        node_states={"n1": ExecutionNodeStatus.COMPLETED.value},
        prior_node_outputs={"n1": {"agent_id": "a1", "summary": "done", "status": "completed"}},
    )
    node = ExecutionNode(node_id="n1", agent_id="a1", status=ExecutionNodeStatus.COMPLETED)
    prior = {
        "n1": AgentExecutionResult(
            agent_id="a1",
            run_id="run_1",
            status=AgentExecutionStatus.COMPLETED,
            summary="done",
        )
    }
    assert should_skip_graph_node(node, checkpoint=ckpt, prior_outputs=prior)
    failed = ExecutionNode(node_id="n2", agent_id="a2", status=ExecutionNodeStatus.FAILED)
    assert not should_skip_graph_node(failed, checkpoint=ckpt, prior_outputs=prior)


@pytest.mark.unit
@pytest.mark.gate
def test_apply_runtime_checkpoint_to_graph_restores_completed_node():
    graph = ExecutionGraph(
        graph_id="g1",
        task_id="task_1",
        nodes=[
            ExecutionNode(node_id="n1", agent_id="a1"),
            ExecutionNode(node_id="n2", agent_id="a2", depends_on=["n1"]),
        ],
    )
    runtime = RuntimeCheckpointExecutionState(
        node_states={
            "n1": ExecutionNodeStatus.COMPLETED.value,
            "n2": ExecutionNodeStatus.FAILED.value,
        },
        prior_node_outputs={
            "n1": {"agent_id": "a1", "summary": "step one", "status": "completed"},
        },
    )
    prior: dict = {}
    apply_runtime_checkpoint_to_graph(graph, runtime, prior)

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

    async def _fake_handle_task(task: Task, *, run_id, attempt_id=None) -> TaskResult:
        captured["run_id"] = run_id
        captured["attempt_id"] = attempt_id
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "handle_task", _fake_handle_task)
    runner = UnifiedTaskRunner(loop)
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="resume")
    checkpoint = TaskCheckpoint(
        task_id=task.task_id,
        tenant_id="t1",
        resume_token="rt_test",
        task_state=TaskState.WAITING_FOR_HUMAN,
        runtime=RuntimeCheckpoint(run_id=run_id, attempt_id=attempt_id),
    )

    await runner.run_task(task, resume_checkpoint=checkpoint)

    assert captured["run_id"] == run_id
    assert captured["attempt_id"] == attempt_id


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_unified_task_runner_resume_rejects_explicit_identity_mismatch(monkeypatch):
    from intergrax.runtime.nexus.nexus_loop import NexusLoop
    from intergrax.runtime.registry.agent_registry import AgentRegistry

    loop = NexusLoop(AgentRegistry())
    checkpoint_run_id = mint_run_id()
    checkpoint_attempt_id = mint_attempt_id()
    runner = UnifiedTaskRunner(loop)
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="resume")
    checkpoint = TaskCheckpoint(
        task_id=task.task_id,
        tenant_id="t1",
        resume_token="rt_test",
        task_state=TaskState.WAITING_FOR_HUMAN,
        runtime=RuntimeCheckpoint(
            run_id=checkpoint_run_id,
            attempt_id=checkpoint_attempt_id,
        ),
    )

    with pytest.raises(ValueError, match="explicit run_id conflicts"):
        await runner.run_task(
            task,
            run_id=mint_run_id(),
            attempt_id=checkpoint_attempt_id,
            resume_checkpoint=checkpoint,
        )
    with pytest.raises(ValueError, match="explicit attempt_id conflicts"):
        await runner.run_task(
            task,
            run_id=checkpoint_run_id,
            attempt_id=mint_attempt_id(),
            resume_checkpoint=checkpoint,
        )


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_resume_metadata_identity_does_not_override_checkpoint(monkeypatch):
    from intergrax.runtime.nexus.nexus_loop import NexusLoop
    from intergrax.runtime.registry.agent_registry import AgentRegistry

    captured: dict[str, object] = {}
    loop = NexusLoop(AgentRegistry())
    checkpoint_run_id = mint_run_id()
    checkpoint_attempt_id = mint_attempt_id()
    metadata_run_id = mint_run_id()
    metadata_attempt_id = mint_attempt_id()

    async def _fake_handle_task(task: Task, *, run_id, attempt_id=None) -> TaskResult:
        captured["run_id"] = run_id
        captured["attempt_id"] = attempt_id
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "handle_task", _fake_handle_task)
    runner = UnifiedTaskRunner(loop)
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="resume")
    task.metadata[RUNTIME_CHECKPOINT_KEY] = {
        "run_id": metadata_run_id,
        "attempt_id": metadata_attempt_id,
        "uaep_step_index": 0,
    }
    checkpoint = TaskCheckpoint(
        task_id=task.task_id,
        tenant_id="t1",
        resume_token="rt_test",
        task_state=TaskState.WAITING_FOR_HUMAN,
        runtime=RuntimeCheckpoint(
            run_id=checkpoint_run_id,
            attempt_id=checkpoint_attempt_id,
        ),
    )

    await runner.run_task(task, resume_checkpoint=checkpoint)

    assert captured["run_id"] == checkpoint_run_id
    assert captured["attempt_id"] == checkpoint_attempt_id
    assert captured["run_id"] != metadata_run_id
    assert captured["attempt_id"] != metadata_attempt_id

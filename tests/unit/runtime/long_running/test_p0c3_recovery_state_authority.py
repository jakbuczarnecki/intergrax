# © Artur Czarnecki. All rights reserved.

"""P0C-3 recovery state authority proofs."""

from __future__ import annotations

import copy

import pytest

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
    prepare_task_for_checkpoint_resume,
    should_skip_graph_node,
)
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    ExecutionCheckpointEntry,
    ExecutionCheckpointStatus,
    ExecutionPriorOutput,
    ExecutionTreeSnapshot,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
from intergrax.runtime.nexus.execution.execution_graph import (
    ExecutionGraph,
    ExecutionNode,
    ExecutionNodeStatus,
)
from intergrax.runtime.task.task import Task, TaskState


def _tree(
    *,
    task_id: str,
    run_id: str,
    attempt_id: str,
    root_execution_id: str,
    extra_entries: list[ExecutionCheckpointEntry] | None = None,
) -> ExecutionTreeSnapshot:
    entries = [
        ExecutionCheckpointEntry(
            execution_id=root_execution_id,
            parent_execution_id=None,
            status=ExecutionCheckpointStatus.RUNNING,
        ),
    ]
    if extra_entries:
        entries.extend(extra_entries)
    return ExecutionTreeSnapshot(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        entries=entries,
    )


def _child_entry(
    execution_id: str,
    *,
    parent: str,
    graph_node_id: str,
    status: ExecutionCheckpointStatus,
    prior_output: ExecutionPriorOutput | None = None,
) -> ExecutionCheckpointEntry:
    return ExecutionCheckpointEntry(
        execution_id=execution_id,
        parent_execution_id=parent,
        status=status,
        graph_node_id=graph_node_id,
        prior_output=prior_output,
    )


def _runtime(
    *,
    task_id: str,
    run_id: str,
    attempt_id: str,
    root: str,
    extra_entries: list[ExecutionCheckpointEntry] | None = None,
    **fields: object,
) -> RuntimeCheckpoint:
    return RuntimeCheckpoint(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_tree=_tree(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            root_execution_id=root,
            extra_entries=extra_entries,
        ),
        **fields,
    )


def _graph(task_id: str, *node_ids: str) -> ExecutionGraph:
    return ExecutionGraph(
        graph_id="g1",
        task_id=task_id,
        nodes=[ExecutionNode(node_id=node_id, agent_id=f"a{index}") for index, node_id in enumerate(node_ids, start=1)],
    )


@pytest.mark.unit
def test_tree_completed_overrides_node_states_failed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    child = mint_execution_id()
    runtime = _runtime(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        root=root,
        extra_entries=[
            _child_entry(
                child,
                parent=root,
                graph_node_id="n1",
                status=ExecutionCheckpointStatus.COMPLETED,
                prior_output=ExecutionPriorOutput(
                    agent_id="a1",
                    summary="canonical",
                    status="completed",
                    graph_node_id="n1",
                ),
            )
        ],
        node_states={"n1": ExecutionNodeStatus.FAILED.value},
    )
    graph = _graph(task_id, "n1")
    prior: dict[str, AgentExecutionResult] = {}
    apply_runtime_checkpoint_to_graph(graph, runtime, prior, run_id=run_id)

    node = graph.node_by_id("n1")
    assert node.status == ExecutionNodeStatus.COMPLETED
    assert prior["n1"].summary == "canonical"
    assert should_skip_graph_node(node, checkpoint=runtime, prior_outputs=prior)


@pytest.mark.unit
def test_tree_failed_not_skipped_when_node_states_completed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    child = mint_execution_id()
    runtime = _runtime(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        root=root,
        extra_entries=[
            _child_entry(
                child,
                parent=root,
                graph_node_id="n1",
                status=ExecutionCheckpointStatus.FAILED,
                prior_output=ExecutionPriorOutput(
                    agent_id="a1",
                    summary="failed-run",
                    status="failed",
                    graph_node_id="n1",
                ),
            )
        ],
        node_states={"n1": ExecutionNodeStatus.COMPLETED.value},
        prior_node_outputs={
            "n1": {
                "agent_id": "a1",
                "summary": "legacy-completed",
                "status": "completed",
            }
        },
    )
    graph = _graph(task_id, "n1")
    prior: dict[str, AgentExecutionResult] = {}
    apply_runtime_checkpoint_to_graph(graph, runtime, prior, run_id=run_id)

    node = graph.node_by_id("n1")
    assert node.status == ExecutionNodeStatus.FAILED
    assert prior["n1"].summary == "failed-run"
    assert should_skip_graph_node(node, checkpoint=runtime, prior_outputs=prior) is False


@pytest.mark.unit
def test_tree_prior_output_wins_over_legacy_prior_node_outputs() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    child = mint_execution_id()
    runtime = _runtime(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        root=root,
        extra_entries=[
            _child_entry(
                child,
                parent=root,
                graph_node_id="n1",
                status=ExecutionCheckpointStatus.COMPLETED,
                prior_output=ExecutionPriorOutput(
                    agent_id="a1",
                    summary="canonical-output",
                    status="completed",
                    graph_node_id="n1",
                ),
            )
        ],
        prior_node_outputs={
            "n1": {
                "agent_id": "a1",
                "summary": "legacy-output",
                "status": "completed",
            }
        },
    )
    graph = _graph(task_id, "n1")
    prior: dict[str, AgentExecutionResult] = {}
    apply_runtime_checkpoint_to_graph(graph, runtime, prior, run_id=run_id)

    assert prior["n1"].summary == "canonical-output"


@pytest.mark.unit
def test_graph_snapshot_lifecycle_overridden_by_tree_failed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    child = mint_execution_id()
    snapshot_graph = ExecutionGraph(
        graph_id="g1",
        task_id=task_id,
        nodes=[
            ExecutionNode(
                node_id="n1",
                agent_id="a1",
                status=ExecutionNodeStatus.COMPLETED,
                execution_result=AgentExecutionResult(
                    agent_id="a1",
                    run_id=run_id,
                    status=AgentExecutionStatus.COMPLETED,
                    summary="snapshot-output",
                ),
            )
        ],
    )
    runtime = _runtime(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        root=root,
        extra_entries=[
            _child_entry(
                child,
                parent=root,
                graph_node_id="n1",
                status=ExecutionCheckpointStatus.FAILED,
                prior_output=ExecutionPriorOutput(
                    agent_id="a1",
                    summary="tree-failed",
                    status="failed",
                    graph_node_id="n1",
                ),
            )
        ],
        graph_snapshot=snapshot_graph.model_dump(mode="json"),
    )
    graph = _graph(task_id, "n1")
    prior: dict[str, AgentExecutionResult] = {}
    apply_runtime_checkpoint_to_graph(graph, runtime, prior, run_id=run_id)

    node = graph.node_by_id("n1")
    assert node.status == ExecutionNodeStatus.FAILED
    assert prior["n1"].summary == "tree-failed"
    assert should_skip_graph_node(node, checkpoint=runtime, prior_outputs=prior) is False


@pytest.mark.unit
def test_legacy_fallback_without_tree_entry() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    runtime = _runtime(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        root=root,
        node_states={"n1": ExecutionNodeStatus.COMPLETED.value},
        prior_node_outputs={
            "n1": {
                "agent_id": "a1",
                "summary": "legacy-only",
                "status": "completed",
            }
        },
    )
    graph = _graph(task_id, "n1")
    prior: dict[str, AgentExecutionResult] = {}
    apply_runtime_checkpoint_to_graph(graph, runtime, prior, run_id=run_id)

    node = graph.node_by_id("n1")
    assert node.status == ExecutionNodeStatus.COMPLETED
    assert prior["n1"].summary == "legacy-only"
    assert should_skip_graph_node(node, checkpoint=runtime, prior_outputs=prior)


@pytest.mark.unit
def test_completed_tree_without_prior_output_fails_closed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    child = mint_execution_id()
    runtime = _runtime(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        root=root,
        extra_entries=[
            _child_entry(
                child,
                parent=root,
                graph_node_id="n1",
                status=ExecutionCheckpointStatus.COMPLETED,
                prior_output=None,
            )
        ],
    )
    graph = _graph(task_id, "n1")
    with pytest.raises(ValueError, match="missing canonical prior_output"):
        apply_runtime_checkpoint_to_graph(graph, runtime, {}, run_id=run_id)


@pytest.mark.unit
def test_build_runtime_checkpoint_replaces_stale_legacy_projections() -> None:
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
        task.runtime.orchestration.runtime_checkpoint = RuntimeCheckpoint(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_tree=_tree(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                root_execution_id=root,
            ),
            node_states={"stale": ExecutionNodeStatus.COMPLETED.value},
            prior_node_outputs={
                "stale": {
                    "agent_id": "a-stale",
                    "summary": "stale",
                    "status": "completed",
                }
            },
            uaep_step_index=3,
            uaep_step_id="review",
        )
        graph = ExecutionGraph(
            graph_id="g1",
            task_id=task_id,
            nodes=[ExecutionNode(node_id="n1", agent_id="a1", status=ExecutionNodeStatus.PENDING)],
        )
        runtime = build_runtime_checkpoint(
            task,
            run_id=run_id,
            attempt_id=attempt_id,
            graph=graph,
        )
    finally:
        reset_active_execution_identity(token)

    assert "stale" not in runtime.node_states
    assert "stale" not in runtime.prior_node_outputs
    assert runtime.node_states["n1"] == ExecutionNodeStatus.PENDING.value
    assert runtime.uaep_step_index == 3
    assert runtime.uaep_step_id == "review"


@pytest.mark.unit
def test_prepare_resume_does_not_mutate_source_checkpoint() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    source_tree = _tree(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        root_execution_id=root,
    )
    source_runtime = RuntimeCheckpoint(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_tree=source_tree,
        node_states={"n1": ExecutionNodeStatus.FAILED.value},
    )
    source_checkpoint = TaskCheckpoint(
        task_id=task_id,
        tenant_id="t1",
        resume_token="rt_test",
        task_state=TaskState.WAITING_FOR_HUMAN,
        runtime=source_runtime,
    )
    original = copy.deepcopy(source_checkpoint.model_dump(mode="json"))
    task = Task(tenant_id="t1", user_id="u1", agent_id="a1", task_id=task_id)
    new_attempt = mint_attempt_id()
    new_root = mint_execution_id()

    prepare_task_for_checkpoint_resume(
        task,
        source_checkpoint,
        active_attempt_id=new_attempt,
        active_root_execution_id=new_root,
    )

    assert source_checkpoint.model_dump(mode="json") == original
    assert task.runtime.orchestration.runtime_checkpoint is not None
    assert task.runtime.orchestration.runtime_checkpoint.attempt_id == new_attempt

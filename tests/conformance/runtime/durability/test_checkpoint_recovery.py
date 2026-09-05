# © Artur Czarnecki. All rights reserved.

"""P0C-2 + P0C-3 — checkpoint resume and recovery state authority."""

from __future__ import annotations

import copy

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_terminal import ExecutionTerminalOutcome
from intergrax.runtime.cancellation.resume_admission import (
    CheckpointNotResumableError,
    assert_checkpoint_resumable,
)
from intergrax.runtime.execution.orchestration import resolve_root_task_identity
from intergrax.runtime.long_running.checkpoint_builder import (
    apply_runtime_checkpoint_to_graph,
    prepare_task_for_checkpoint_resume,
)
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    ExecutionCheckpointEntry,
    ExecutionCheckpointStatus,
    ExecutionPriorOutput,
    ExecutionTreeSnapshot,
    minimal_runtime_checkpoint,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.nexus.execution.execution_graph import (
    ExecutionGraph,
    ExecutionNode,
    ExecutionNodeStatus,
)
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions

from tests.conformance.runtime.durability.provider_factories import create_checkpoint_store
from tests.conformance.runtime.durability.restart import fresh_checkpoint_composition

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-p0c8-ckpt"


def _paused_checkpoint(*, task_id: str | None = None) -> TaskCheckpoint:
    resolved_task_id = task_id or str(mint_task_id())
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    task = Task(
        task_id=resolved_task_id,
        tenant_id=_TENANT,
        user_id="user",
        message="paused",
        state=TaskState.WAITING_FOR_HUMAN,
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, resume_token="rt-p0c8"),
        ),
    )
    return TaskCheckpoint(
        task_id=resolved_task_id,
        tenant_id=_TENANT,
        resume_token="rt-p0c8",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=task.model_dump(mode="json"),
        runtime=minimal_runtime_checkpoint(
            task_id=resolved_task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            root_execution_id=mint_execution_id(),
        ),
    )


def test_checkpoint_resume_identity_survives_restart(tmp_path) -> None:
    db_path = tmp_path / "checkpoint-resume.db"
    store_a = create_checkpoint_store(db_path)
    checkpoint = _paused_checkpoint()
    store_a.save(checkpoint)

    store_b = create_checkpoint_store(db_path)
    loaded = store_b.get_by_token(checkpoint.task_id, _TENANT, checkpoint.resume_token)
    assert loaded is not None and loaded.runtime is not None

    resumed_root = mint_execution_id()
    identity = resolve_root_task_identity(resume_checkpoint=loaded)
    assert identity.run_id == loaded.runtime.run_id
    assert identity.attempt_id == loaded.runtime.attempt_id

    task = Task(
        task_id=checkpoint.task_id,
        tenant_id=_TENANT,
        user_id="user",
        message="resume",
    )
    runtime = prepare_task_for_checkpoint_resume(
        task,
        loaded,
        active_attempt_id=loaded.runtime.attempt_id,
        active_root_execution_id=resumed_root,
    )
    assert runtime.run_id == loaded.runtime.run_id
    assert runtime.attempt_id == loaded.runtime.attempt_id
    assert resumed_root != loaded.runtime.execution_tree.entries[0].execution_id


def test_checkpoint_explicit_attempt_conflict_denies_after_restart(tmp_path) -> None:
    checkpoint = _paused_checkpoint()
    assert checkpoint.runtime is not None
    incoming_a2 = mint_attempt_id()

    with pytest.raises(ValueError, match="explicit attempt_id conflicts"):
        resolve_root_task_identity(
            attempt_id=incoming_a2,
            resume_checkpoint=checkpoint,
        )

    db_path = tmp_path / "checkpoint-conflict.db"
    store_a = create_checkpoint_store(db_path)
    store_a.save(checkpoint)
    store_b = create_checkpoint_store(db_path)
    loaded = store_b.get_by_token(checkpoint.task_id, _TENANT, checkpoint.resume_token)
    assert loaded is not None

    with pytest.raises(ValueError, match="explicit attempt_id conflicts"):
        resolve_root_task_identity(
            attempt_id=incoming_a2,
            resume_checkpoint=loaded,
        )


def test_recovery_precedence_survives_restart() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    child = mint_execution_id()
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
                    graph_node_id="n1",
                    status=ExecutionCheckpointStatus.COMPLETED,
                    prior_output=ExecutionPriorOutput(
                        agent_id="a1",
                        summary="tree-canonical",
                        status="completed",
                        graph_node_id="n1",
                    ),
                ),
            ],
        ),
        node_states={"n1": ExecutionNodeStatus.FAILED.value},
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="a1")],
    )
    prior: dict[str, AgentExecutionResult] = {}
    apply_runtime_checkpoint_to_graph(graph, runtime, prior, run_id=run_id)

    restarted_runtime = copy.deepcopy(runtime)
    restarted_graph = ExecutionGraph(
        graph_id="g1",
        task_id=task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="a1")],
    )
    restarted_prior: dict[str, AgentExecutionResult] = {}
    apply_runtime_checkpoint_to_graph(
        restarted_graph,
        restarted_runtime,
        restarted_prior,
        run_id=run_id,
    )

    node = restarted_graph.node_by_id("n1")
    assert node.status == ExecutionNodeStatus.COMPLETED
    assert restarted_prior["n1"].summary == "tree-canonical"


def test_recovery_corruption_fails_closed_after_restart() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    child = mint_execution_id()
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
                    graph_node_id="n1",
                    status=ExecutionCheckpointStatus.COMPLETED,
                    prior_output=None,
                ),
            ],
        ),
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="a1")],
    )
    with pytest.raises(ValueError, match="missing canonical prior_output"):
        apply_runtime_checkpoint_to_graph(graph, runtime, {}, run_id=run_id)


def test_resume_plus_terminal_race_aftermath_denies_checkpoint(tmp_path) -> None:
    db_path = tmp_path / "resume-terminal.db"
    store_a, terminal_a = fresh_checkpoint_composition(db_path)
    checkpoint = _paused_checkpoint()
    store_a.save(checkpoint)
    assert checkpoint.runtime is not None
    terminal_a.commit_terminal_outcome(
        tenant_id=checkpoint.tenant_id,
        task_id=checkpoint.task_id,
        run_id=checkpoint.runtime.run_id,
        outcome=ExecutionTerminalOutcome.COMPLETED,
        reason="completed",
    )

    store_b = create_checkpoint_store(db_path)
    _, terminal_b = fresh_checkpoint_composition(db_path)
    loaded = store_b.get_by_token(checkpoint.task_id, _TENANT, checkpoint.resume_token)
    assert loaded is not None

    with pytest.raises(CheckpointNotResumableError):
        assert_checkpoint_resumable(loaded, execution_terminal=terminal_b)

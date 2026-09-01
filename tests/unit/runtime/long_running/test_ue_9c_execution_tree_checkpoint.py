# © Artur Czarnecki. All rights reserved.

"""UE-9C — canonical Execution Tree checkpoint and resume."""

from __future__ import annotations

import json
from pathlib import Path

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
    prepare_task_for_checkpoint_resume,
    should_skip_graph_node,
    sync_execution_tree_to_task,
)
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    ExecutionCheckpointEntry,
    ExecutionCheckpointStatus,
    ExecutionPriorOutput,
    ExecutionTreeRecorder,
    ExecutionTreeSnapshot,
    build_execution_tree_resume_plan,
    minimal_execution_tree_snapshot,
    minimal_runtime_checkpoint,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode, ExecutionNodeStatus
from intergrax.runtime.task.task import Task, TaskState

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _entry(
    execution_id: str,
    *,
    parent: str | None = None,
    status: ExecutionCheckpointStatus = ExecutionCheckpointStatus.COMPLETED,
    graph_node_id: str | None = None,
    resumed_from: str | None = None,
    prior_output: ExecutionPriorOutput | None = None,
) -> ExecutionCheckpointEntry:
    return ExecutionCheckpointEntry(
        execution_id=execution_id,
        parent_execution_id=parent,
        status=status,
        graph_node_id=graph_node_id,
        resumed_from_execution_id=resumed_from,
        prior_output=prior_output,
    )


def _tree(
    *,
    task_id: str | None = None,
    run_id: str | None = None,
    attempt_id: str | None = None,
    entries: list[ExecutionCheckpointEntry],
) -> ExecutionTreeSnapshot:
    return ExecutionTreeSnapshot(
        task_id=task_id or mint_task_id(),
        run_id=run_id or mint_run_id(),
        attempt_id=attempt_id or mint_attempt_id(),
        entries=entries,
    )


def _runtime(
    *,
    task_id: str | None = None,
    run_id: str | None = None,
    attempt_id: str | None = None,
    entries: list[ExecutionCheckpointEntry],
) -> RuntimeCheckpoint:
    resolved_task_id = task_id or mint_task_id()
    resolved_run_id = run_id or mint_run_id()
    resolved_attempt_id = attempt_id or mint_attempt_id()
    return RuntimeCheckpoint(
        run_id=resolved_run_id,
        attempt_id=resolved_attempt_id,
        execution_tree=_tree(
            task_id=resolved_task_id,
            run_id=resolved_run_id,
            attempt_id=resolved_attempt_id,
            entries=entries,
        ),
    )


def test_root_entry_has_no_parent() -> None:
    root_id = mint_execution_id()
    tree = _tree(entries=[_entry(root_id, status=ExecutionCheckpointStatus.RUNNING)])
    tree.validate_tree()
    assert tree.entries[0].parent_execution_id is None


def test_child_parent_link() -> None:
    root_id = mint_execution_id()
    child_id = mint_execution_id()
    tree = _tree(
        entries=[
            _entry(root_id, status=ExecutionCheckpointStatus.RUNNING),
            _entry(child_id, parent=root_id, graph_node_id="n1"),
        ]
    )
    tree.validate_tree()
    assert tree.entry_by_graph_node_id("n1").parent_execution_id == root_id


def test_nested_parent_chain() -> None:
    e1 = mint_execution_id()
    e2 = mint_execution_id()
    e3 = mint_execution_id()
    tree = _tree(
        entries=[
            _entry(e1, status=ExecutionCheckpointStatus.RUNNING),
            _entry(e2, parent=e1, graph_node_id="n1"),
            _entry(e3, parent=e2, graph_node_id="n2"),
        ]
    )
    tree.validate_tree()
    assert tree.entry_by_graph_node_id("n2").parent_execution_id == e2


def test_parallel_children_preserved() -> None:
    root = mint_execution_id()
    e2 = mint_execution_id()
    e3 = mint_execution_id()
    e4 = mint_execution_id()
    tree = _tree(
        entries=[
            _entry(root, status=ExecutionCheckpointStatus.RUNNING),
            _entry(
                e2,
                parent=root,
                graph_node_id="n2",
                prior_output=ExecutionPriorOutput(
                    agent_id="a2",
                    summary="done-2",
                    status="completed",
                    graph_node_id="n2",
                ),
            ),
            _entry(e3, parent=root, graph_node_id="n3", status=ExecutionCheckpointStatus.INTERRUPTED),
            _entry(
                e4,
                parent=root,
                graph_node_id="n4",
                prior_output=ExecutionPriorOutput(
                    agent_id="a4",
                    summary="done-4",
                    status="completed",
                    graph_node_id="n4",
                ),
            ),
        ]
    )
    tree.validate_tree()
    assert tree.completed_graph_node_ids() == frozenset({"n2", "n4"})


def test_completed_child_skipped_on_resume() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    completed_child = mint_execution_id()
    runtime = _runtime(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        entries=[
            _entry(root, status=ExecutionCheckpointStatus.RUNNING),
            _entry(
                completed_child,
                parent=root,
                graph_node_id="n1",
                prior_output=ExecutionPriorOutput(
                    agent_id="a1",
                    summary="cached",
                    status="completed",
                    graph_node_id="n1",
                ),
            ),
        ],
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="a1")],
    )
    prior: dict[str, AgentExecutionResult] = {}
    apply_runtime_checkpoint_to_graph(graph, runtime, prior, run_id=run_id)
    node = graph.node_by_id("n1")
    assert should_skip_graph_node(node, checkpoint=runtime, prior_outputs=prior)


def test_incomplete_child_not_marked_completed() -> None:
    interrupted = mint_execution_id()
    runtime = _runtime(
        entries=[
            _entry(mint_execution_id(), status=ExecutionCheckpointStatus.RUNNING),
            _entry(interrupted, parent=mint_execution_id(), graph_node_id="n1", status=ExecutionCheckpointStatus.INTERRUPTED),
        ]
    )
    with pytest.raises(ValueError, match="parent missing"):
        runtime.validate_canonical()


def test_redelivery_mints_new_active_execution_id() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    root_a1 = mint_execution_id()
    interrupted = mint_execution_id()
    checkpoint_tree = _tree(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
        entries=[
            _entry(root_a1, status=ExecutionCheckpointStatus.RUNNING),
            _entry(interrupted, parent=root_a1, graph_node_id="n3", status=ExecutionCheckpointStatus.INTERRUPTED),
        ],
    )
    attempt_a2 = mint_attempt_id()
    root_a2 = mint_execution_id()
    plan = build_execution_tree_resume_plan(
        checkpoint_tree,
        task_id=task_id,
        run_id=run_id,
        new_attempt_id=attempt_a2,
        new_root_execution_id=root_a2,
    )
    assert plan.active_snapshot.attempt_id == attempt_a2
    resumed = [
        entry
        for entry in plan.active_snapshot.entries
        if entry.resumed_from_execution_id == interrupted
    ]
    assert len(resumed) == 0
    assert plan.resume_graph_node_ids == frozenset({"n3"})
    historical_n3 = plan.historical_by_graph_node_id["n3"]
    assert historical_n3.execution_id == interrupted


def test_resume_plan_retains_historical_lookup_for_failed_child() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    root_a1 = mint_execution_id()
    failed_child = mint_execution_id()
    checkpoint_tree = _tree(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
        entries=[
            _entry(root_a1, status=ExecutionCheckpointStatus.RUNNING),
            _entry(
                failed_child,
                parent=root_a1,
                graph_node_id="n_c",
                status=ExecutionCheckpointStatus.FAILED,
            ),
        ],
    )
    plan = build_execution_tree_resume_plan(
        checkpoint_tree,
        task_id=task_id,
        run_id=run_id,
        new_attempt_id=attempt_a1,
        new_root_execution_id=mint_execution_id(),
    )
    assert plan.resume_graph_node_ids == frozenset({"n_c"})
    assert plan.historical_by_graph_node_id["n_c"].execution_id == failed_child


def test_resume_plan_reparents_nested_completed_children() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    old_root = mint_execution_id()
    execution_a = mint_execution_id()
    execution_b = mint_execution_id()
    prior_a = ExecutionPriorOutput(
        agent_id="a1",
        summary="done-a",
        status="completed",
        graph_node_id="n_a",
    )
    prior_b = ExecutionPriorOutput(
        agent_id="a2",
        summary="done-b",
        status="completed",
        graph_node_id="n_b",
    )
    checkpoint_tree = _tree(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
        entries=[
            _entry(old_root, status=ExecutionCheckpointStatus.INTERRUPTED),
            _entry(
                execution_a,
                parent=old_root,
                graph_node_id="n_a",
                prior_output=prior_a,
            ),
            _entry(
                execution_b,
                parent=execution_a,
                graph_node_id="n_b",
                prior_output=prior_b,
            ),
        ],
    )
    attempt_a2 = mint_attempt_id()
    new_root = mint_execution_id()
    plan = build_execution_tree_resume_plan(
        checkpoint_tree,
        task_id=task_id,
        run_id=run_id,
        new_attempt_id=attempt_a2,
        new_root_execution_id=new_root,
    )

    assert plan.historical_snapshot != plan.active_snapshot
    assert new_root != old_root

    historical_root = next(
        entry
        for entry in plan.historical_snapshot.entries
        if entry.parent_execution_id is None
    )
    assert historical_root.execution_id == old_root

    active_root = next(
        entry
        for entry in plan.active_snapshot.entries
        if entry.parent_execution_id is None
    )
    assert active_root.execution_id == new_root
    assert active_root.parent_execution_id is None

    historical_a = plan.historical_snapshot.entry_by_graph_node_id("n_a")
    historical_b = plan.historical_snapshot.entry_by_graph_node_id("n_b")
    assert historical_a is not None
    assert historical_b is not None
    assert historical_a.parent_execution_id == old_root
    assert historical_b.parent_execution_id == execution_a

    active_a = plan.active_snapshot.entry_by_graph_node_id("n_a")
    active_b = plan.active_snapshot.entry_by_graph_node_id("n_b")
    assert active_a is not None
    assert active_b is not None
    assert active_a.execution_id == execution_a
    assert active_b.execution_id == execution_b
    assert active_a.parent_execution_id == new_root
    assert active_b.parent_execution_id == execution_a

    active_execution_ids = {
        entry.execution_id for entry in plan.active_snapshot.entries
    }
    assert old_root not in active_execution_ids
    for entry in plan.active_snapshot.entries:
        if entry.parent_execution_id is not None:
            assert entry.parent_execution_id != old_root
            assert entry.parent_execution_id in active_execution_ids

    plan.historical_snapshot.validate_tree()
    plan.active_snapshot.validate_tree()
    assert plan.skip_graph_node_ids == frozenset({"n_a", "n_b"})


def test_old_incomplete_execution_not_reused_as_active() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    root_a1 = mint_execution_id()
    old_child = mint_execution_id()
    checkpoint_tree = _tree(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
        entries=[
            _entry(root_a1, status=ExecutionCheckpointStatus.INTERRUPTED),
            _entry(old_child, parent=root_a1, graph_node_id="n1", status=ExecutionCheckpointStatus.INTERRUPTED),
        ],
    )
    task = Task(tenant_id="t1", user_id="u1", agent_id="a1", task_id=task_id)
    checkpoint = TaskCheckpoint(
        task_id=task_id,
        tenant_id="t1",
        resume_token="rt",
        task_state=TaskState.WAITING_FOR_HUMAN,
        runtime=RuntimeCheckpoint(
            run_id=run_id,
            attempt_id=attempt_a1,
            execution_tree=checkpoint_tree,
        ),
    )
    attempt_a2 = mint_attempt_id()
    root_a2 = mint_execution_id()
    runtime, _resume_plan = prepare_task_for_checkpoint_resume(
        task,
        checkpoint,
        active_attempt_id=attempt_a2,
        active_root_execution_id=root_a2,
    )
    active_ids = {entry.execution_id for entry in runtime.execution_tree.entries}
    assert old_child not in active_ids
    assert root_a2 in active_ids


def test_tree_parent_missing_fail_closed() -> None:
    tree = _tree(
        entries=[
            _entry(mint_execution_id(), status=ExecutionCheckpointStatus.RUNNING),
            _entry(mint_execution_id(), parent=mint_execution_id(), graph_node_id="n1"),
        ]
    )
    with pytest.raises(ValueError, match="parent missing"):
        tree.validate_tree()


def test_tree_cycle_fail_closed() -> None:
    e1 = mint_execution_id()
    e2 = mint_execution_id()
    tree = _tree(
        entries=[
            _entry(e1, parent=e2, status=ExecutionCheckpointStatus.RUNNING),
            _entry(e2, parent=e1, graph_node_id="n1"),
        ]
    )
    with pytest.raises(ValueError, match="exactly one root"):
        tree.validate_tree()


def test_wrong_run_id_fail_closed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    other_run = mint_run_id()
    runtime = _runtime(
        task_id=task_id,
        run_id=run_id,
        entries=[_entry(mint_execution_id(), status=ExecutionCheckpointStatus.RUNNING)],
    )
    with pytest.raises(ValueError, match="run_id mismatch"):
        runtime.execution_tree.validate_for_task(task_id=task_id, run_id=other_run)


def test_corrupt_execution_id_fail_closed() -> None:
    with pytest.raises(ValidationError):
        ExecutionCheckpointEntry(
            execution_id="not-an-exec-id",
            parent_execution_id=None,
            status=ExecutionCheckpointStatus.RUNNING,
        )


def test_no_task_run_id_placeholder_in_repo() -> None:
    repo_root = Path(__file__).resolve().parents[4] / "intergrax"
    hits: list[str] = []
    for path in repo_root.rglob("*.py"):
        rel = path.relative_to(repo_root.parent).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "task_run_id_placeholder" in text:
            hits.append(rel)
    assert hits == []


def test_no_metadata_checkpoint_authority_symbols() -> None:
    repo_root = Path(__file__).resolve().parents[4] / "intergrax"
    forbidden = (
        "RUNTIME_CHECKPOINT_KEY",
        "runtime_checkpoint_from_metadata",
        "runtime_checkpoint_from_execution_structured",
        "attach_runtime_checkpoint_to_metadata",
        "RuntimeCheckpointExecutionState",
    )
    hits: list[str] = []
    for path in repo_root.rglob("*.py"):
        rel = path.relative_to(repo_root.parent).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for symbol in forbidden:
            if symbol in text:
                hits.append(f"{rel}:{symbol}")
    assert hits == []


def test_checkpoint_survives_store_roundtrip(tmp_path: Path) -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    runtime = _runtime(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        entries=[_entry(root, status=ExecutionCheckpointStatus.RUNNING)],
    )
    task = Task(tenant_id="t1", user_id="u1", agent_id="a1", task_id=task_id)
    checkpoint = TaskCheckpoint(
        task_id=task_id,
        tenant_id="t1",
        resume_token="rt",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=task.model_dump(mode="json"),
        runtime=runtime,
    )
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "checkpoints.db")
    store.save(checkpoint)
    loaded = store.get_by_token(task_id, "t1", "rt")
    assert loaded is not None
    assert loaded.runtime is not None
    assert loaded.runtime.execution_tree.entries[0].execution_id == root
    payload = json.loads(json.dumps(loaded.model_dump(mode="json")))
    restored = TaskCheckpoint.model_validate(payload)
    assert restored.runtime is not None
    restored.runtime.validate_canonical()


def test_recorder_sync_updates_task_runtime_checkpoint() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    runtime = _runtime(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        entries=[_entry(root, status=ExecutionCheckpointStatus.RUNNING)],
    )
    task = Task(tenant_id="t1", user_id="u1", agent_id="a1", task_id=task_id)
    task.runtime.orchestration.runtime_checkpoint = runtime
    recorder = ExecutionTreeRecorder.from_snapshot(runtime.execution_tree)
    child = mint_execution_id()
    recorder.record_child_started(
        execution_id=child,
        parent_execution_id=root,
        graph_node_id="n1",
    )
    sync_execution_tree_to_task(task, recorder)
    assert task.runtime.orchestration.runtime_checkpoint is not None
    assert task.runtime.orchestration.runtime_checkpoint.execution_tree.entry_by_graph_node_id("n1") is not None


def test_build_runtime_checkpoint_uses_active_execution_id() -> None:
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
        runtime = build_runtime_checkpoint(
            task,
            run_id=run_id,
            attempt_id=attempt_id,
        )
    finally:
        reset_active_execution_identity(token)
    assert runtime.execution_tree.entries[0].execution_id == root


def test_build_runtime_checkpoint_fail_closed_without_active_execution() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    task = Task(tenant_id="t1", user_id="u1", agent_id="a1", task_id=task_id)
    with pytest.raises(RuntimeError, match="active ExecutionId required"):
        build_runtime_checkpoint(
            task,
            run_id=run_id,
            attempt_id=attempt_id,
        )


def test_build_runtime_checkpoint_existing_tree_without_active_execution() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    existing_tree = _tree(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        entries=[_entry(root, status=ExecutionCheckpointStatus.RUNNING)],
    )
    task = Task(tenant_id="t1", user_id="u1", agent_id="a1", task_id=task_id)
    runtime = build_runtime_checkpoint(
        task,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_tree=existing_tree,
    )
    assert runtime.execution_tree.entries[0].execution_id == root


def test_minimal_execution_tree_snapshot_requires_root_execution_id() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    tree = minimal_execution_tree_snapshot(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        root_execution_id=root,
    )
    assert tree.entries[0].execution_id == root


def test_minimal_runtime_checkpoint_requires_root_execution_id() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    runtime = minimal_runtime_checkpoint(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        root_execution_id=root,
    )
    assert runtime.execution_tree.entries[0].execution_id == root


def test_execution_tree_recorder_start_root_does_not_mint() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    recorder = ExecutionTreeRecorder.start_root(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        root_execution_id=root,
    )
    assert recorder.snapshot.entries[0].execution_id == root


def test_resume_child_links_resumed_from_when_lifecycle_supplies_ids() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a2 = mint_attempt_id()
    root_e6 = mint_execution_id()
    interrupted_e4 = mint_execution_id()
    recorder = ExecutionTreeRecorder.start_root(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a2,
        root_execution_id=root_e6,
    )
    recorder.record_child_started(
        execution_id=mint_execution_id(),
        parent_execution_id=root_e6,
        graph_node_id="n3",
        resumed_from_execution_id=interrupted_e4,
    )
    entry = recorder.snapshot.entry_by_graph_node_id("n3")
    assert entry is not None
    assert entry.resumed_from_execution_id == interrupted_e4

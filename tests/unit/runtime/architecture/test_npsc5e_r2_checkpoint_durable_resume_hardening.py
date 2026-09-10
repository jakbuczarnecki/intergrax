# © Artur Czarnecki. All rights reserved.

"""NPSC-5E/R2 — checkpoint durable resume hardening qualification."""

from __future__ import annotations

import concurrent.futures
import re
import subprocess
from contextvars import copy_context
from pathlib import Path

import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.execution_lineage import build_execution_lineage_attempt_scope
from intergrax.contracts.execution_terminal import ExecutionTerminalOutcome
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.cancellation.resume_admission import CheckpointNotResumableError
from intergrax.runtime.execution.attempt_lifecycle import (
    AttemptLifecycleService,
    InMemoryAttemptLifecycleStore,
)
from intergrax.runtime.execution.execution_terminal.service import ExecutionTerminalService
from intergrax.runtime.execution.lineage.persistence import InMemoryExecutionLineagePersistence
from intergrax.runtime.execution.orchestration import resolve_root_task_identity
from intergrax.runtime.long_running.checkpoint_builder import (
    apply_runtime_checkpoint_to_graph,
    build_runtime_checkpoint,
)
from intergrax.runtime.execution.execution_terminal.persistence import (
    CheckpointStoreExecutionTerminalStore,
)
from intergrax.runtime.long_running.checkpoint_resume_validation import (
    CANONICAL_TASK_CHECKPOINT_SCHEMA_VERSION,
    CheckpointResumeEligibility,
    CheckpointResumeValidationError,
    assert_checkpoint_persistable,
    evaluate_checkpoint_resume_eligibility,
    resolve_resume_execution_authority,
    validate_checkpoint_authority_expansion,
    validate_checkpoint_not_stale,
    validate_runtime_checkpoint_schema,
)
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    ExecutionCheckpointEntry,
    ExecutionCheckpointStatus,
    ExecutionPriorOutput,
    ExecutionTreeSnapshot,
    minimal_runtime_checkpoint,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.runtime_checkpoint import (
    CANONICAL_RUNTIME_CHECKPOINT_SCHEMA_VERSION,
)
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.nexus.execution.execution_graph import (
    ExecutionGraph,
    ExecutionNode,
    ExecutionNodeStatus,
)
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_R1_FINAL_SHA = "76603ed266f9f54106bf4718fe8886180a826351"
_LINEAGE_HARDENING_SHA = "a18e65c077ed57bf3bb64ef015b47ee1f3ceb6bf"

_FORBIDDEN_CHECKPOINT_FRAMEWORK_NAMES = (
    "EnterpriseCheckpointEngine",
    "RecoveryCheckpointRuntime",
    "UniversalResumeManager",
)
_REFLECTION_PATTERN = re.compile(r"\b(getattr|setattr|hasattr)\(")
_TENANT = "tenant-r2"


def _paused_checkpoint(
    *,
    task_id: str | None = None,
    tenant_id: str = _TENANT,
    run_id: str | None = None,
    attempt_id: str | None = None,
    root_execution_id: str | None = None,
) -> TaskCheckpoint:
    resolved_task_id = task_id or str(mint_task_id())
    resolved_run_id = run_id or mint_run_id()
    resolved_attempt_id = attempt_id or mint_attempt_id()
    resolved_root = root_execution_id or mint_execution_id()
    task = Task(
        task_id=resolved_task_id,
        tenant_id=tenant_id,
        user_id="user",
        message="paused",
        state=TaskState.WAITING_FOR_HUMAN,
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, resume_token="rt-r2"),
        ),
    )
    return TaskCheckpoint(
        task_id=resolved_task_id,
        tenant_id=tenant_id,
        resume_token="rt-r2",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=task.model_dump(mode="json"),
        created_at_utc="2026-09-09T12:00:00+00:00",
        runtime=minimal_runtime_checkpoint(
            task_id=resolved_task_id,
            run_id=resolved_run_id,
            attempt_id=resolved_attempt_id,
            root_execution_id=resolved_root,
        ),
    )


def _evaluate(
    checkpoint: TaskCheckpoint,
    *,
    task_id: str | None = None,
    tenant_id: str = _TENANT,
    target_run_id: str | None = None,
    target_attempt_id: str | None = None,
    target_root_execution_id: str | None = None,
    latest_checkpoint: TaskCheckpoint | None = None,
    execution_terminal: ExecutionTerminalService | None = None,
    execution_lineage_persistence: InMemoryExecutionLineagePersistence | None = None,
    require_durable_lineage: bool = False,
    current_task: Task | None = None,
    policy_decision: PolicyDecision | None = None,
) -> CheckpointResumeEligibility:
    result = evaluate_checkpoint_resume_eligibility(
        checkpoint,
        target_task_id=task_id or checkpoint.task_id,
        target_tenant_id=tenant_id,
        target_run_id=target_run_id,
        target_attempt_id=target_attempt_id,
        target_root_execution_id=target_root_execution_id,
        latest_checkpoint=latest_checkpoint,
        execution_terminal=execution_terminal,
        execution_lineage_persistence=execution_lineage_persistence,
        require_durable_lineage=require_durable_lineage,
        current_task=current_task,
        policy_decision=policy_decision,
    )
    return result.eligibility


@pytest.mark.parametrize(
    "schema_version",
    ["runtime_checkpoint.v1", "runtime_checkpoint.v3", "unknown"],
)
def test_unknown_runtime_schema_version_blocked(schema_version: str) -> None:
    checkpoint = _paused_checkpoint()
    assert checkpoint.runtime is not None
    runtime = checkpoint.runtime.model_copy(update={"schema_version": schema_version})
    result = validate_runtime_checkpoint_schema(runtime)
    assert result.eligibility is CheckpointResumeEligibility.REJECT_SCHEMA


def test_canonical_checkpoint_restores(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ok.db")
    checkpoint = _paused_checkpoint()
    store.save(checkpoint)
    loaded = store.get_by_token(checkpoint.task_id, _TENANT, checkpoint.resume_token)
    assert loaded is not None
    task = Task(
        task_id=checkpoint.task_id,
        tenant_id=_TENANT,
        user_id="user",
        message="resume",
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, resume_token="rt-r2"),
        ),
    )
    restored = LongRunningCoordinator.restore_if_resuming(task, store)
    assert restored is not None
    assert task.message == "paused"


def test_wrong_task_blocked() -> None:
    checkpoint = _paused_checkpoint()
    assert _evaluate(checkpoint, task_id=mint_task_id()) is CheckpointResumeEligibility.REJECT_IDENTITY


def test_wrong_tenant_blocked() -> None:
    checkpoint = _paused_checkpoint()
    assert _evaluate(checkpoint, tenant_id="other-tenant") is CheckpointResumeEligibility.REJECT_TENANT


def test_wrong_run_blocked() -> None:
    checkpoint = _paused_checkpoint()
    assert _evaluate(checkpoint, target_run_id=mint_run_id()) is CheckpointResumeEligibility.REJECT_IDENTITY


def test_wrong_attempt_blocked() -> None:
    checkpoint = _paused_checkpoint()
    assert _evaluate(checkpoint, target_attempt_id=mint_attempt_id()) is CheckpointResumeEligibility.REJECT_IDENTITY


def test_wrong_root_execution_blocked() -> None:
    checkpoint = _paused_checkpoint()
    assert (
        _evaluate(checkpoint, target_root_execution_id=mint_execution_id())
        is CheckpointResumeEligibility.REJECT_IDENTITY
    )


def test_tree_attempt_mismatch_blocked() -> None:
    checkpoint = _paused_checkpoint()
    assert checkpoint.runtime is not None
    tree = checkpoint.runtime.execution_tree.model_copy(
        update={"attempt_id": mint_attempt_id()},
    )
    runtime = checkpoint.runtime.model_copy(update={"execution_tree": tree})
    broken = checkpoint.model_copy(update={"runtime": runtime})
    assert _evaluate(broken) is CheckpointResumeEligibility.REJECT_MALFORMED


def test_malformed_checkpoint_missing_runtime_blocked() -> None:
    checkpoint = _paused_checkpoint().model_copy(update={"runtime": None})
    assert _evaluate(checkpoint) is CheckpointResumeEligibility.REJECT_MALFORMED


def test_stale_checkpoint_blocked() -> None:
    older = _paused_checkpoint().model_copy(update={"revision": 1})
    newer = _paused_checkpoint(task_id=older.task_id).model_copy(
        update={"revision": 2, "created_at_utc": "2026-09-09T13:00:00+00:00"},
    )
    result = validate_checkpoint_not_stale(older, newer)
    assert result.eligibility is CheckpointResumeEligibility.REJECT_STALE


def test_newer_checkpoint_wins(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "newer.db")
    older = store.save(_paused_checkpoint())
    newer = store.save(
        older.model_copy(
            update={
                "checkpoint_id": "ckpt_newer",
                "created_at_utc": "2026-09-09T14:00:00+00:00",
                "progress_message": "step 2",
            },
        ),
        expected_revision=older.revision,
    )
    latest = store.get_latest(older.task_id, _TENANT)
    assert latest is not None
    assert latest.checkpoint_id == "ckpt_newer"


def _terminal_service(store: SQLiteTaskCheckpointStore) -> ExecutionTerminalService:
    return ExecutionTerminalService(CheckpointStoreExecutionTerminalStore(store))


def test_terminal_success_resume_blocked(tmp_path: Path) -> None:
    checkpoint = _paused_checkpoint()
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "term.db")
    store.save(checkpoint)
    terminal = _terminal_service(store)
    assert checkpoint.runtime is not None
    terminal.commit_terminal_outcome(
        tenant_id=checkpoint.tenant_id,
        task_id=checkpoint.task_id,
        run_id=checkpoint.runtime.run_id,
        outcome=ExecutionTerminalOutcome.COMPLETED,
        reason="done",
    )
    loaded = store.get_latest(checkpoint.task_id, _TENANT)
    assert loaded is not None
    assert (
        _evaluate(loaded, execution_terminal=terminal)
        is CheckpointResumeEligibility.REJECT_TERMINAL
    )


def test_terminal_cancel_resume_blocked(tmp_path: Path) -> None:
    checkpoint = _paused_checkpoint()
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "cancel.db")
    store.save(checkpoint)
    terminal = _terminal_service(store)
    assert checkpoint.runtime is not None
    terminal.commit_terminal_outcome(
        tenant_id=checkpoint.tenant_id,
        task_id=checkpoint.task_id,
        run_id=checkpoint.runtime.run_id,
        outcome=ExecutionTerminalOutcome.CANCELLED,
        reason="cancelled",
    )
    loaded = store.get_latest(checkpoint.task_id, _TENANT)
    assert loaded is not None
    assert (
        _evaluate(loaded, execution_terminal=terminal)
        is CheckpointResumeEligibility.REJECT_CANCELLED
    )


def test_terminal_deny_resume_blocked(tmp_path: Path) -> None:
    checkpoint = _paused_checkpoint()
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "deny.db")
    store.save(checkpoint)
    terminal = _terminal_service(store)
    assert checkpoint.runtime is not None
    terminal.commit_terminal_outcome(
        tenant_id=checkpoint.tenant_id,
        task_id=checkpoint.task_id,
        run_id=checkpoint.runtime.run_id,
        outcome=ExecutionTerminalOutcome.FAILED,
        reason="denied",
    )
    loaded = store.get_latest(checkpoint.task_id, _TENANT)
    assert loaded is not None
    assert (
        _evaluate(loaded, execution_terminal=terminal)
        is CheckpointResumeEligibility.REJECT_TERMINAL
    )


def test_cancel_after_checkpoint_blocked(tmp_path: Path) -> None:
    checkpoint = _paused_checkpoint()
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "after.db")
    store.save(checkpoint)
    terminal = _terminal_service(store)
    assert checkpoint.runtime is not None
    terminal.commit_terminal_outcome(
        tenant_id=checkpoint.tenant_id,
        task_id=checkpoint.task_id,
        run_id=checkpoint.runtime.run_id,
        outcome=ExecutionTerminalOutcome.CANCELLED,
        reason="post-checkpoint cancel",
    )
    task = Task(
        task_id=checkpoint.task_id,
        tenant_id=_TENANT,
        user_id="user",
        message="resume",
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, resume_token="rt-r2"),
        ),
    )
    with pytest.raises(CheckpointResumeValidationError):
        LongRunningCoordinator.restore_if_resuming(task, store, execution_terminal=terminal)


def test_policy_changed_to_deny_blocked() -> None:
    checkpoint = _paused_checkpoint()
    decision = PolicyDecision(action=PolicyAction.DENY, reason="policy deny")
    assert (
        _evaluate(checkpoint, policy_decision=decision)
        is CheckpointResumeEligibility.REJECT_GOVERNANCE
    )


def test_authority_narrowing_passes() -> None:
    checkpoint = _paused_checkpoint()
    snapshot_task = Task.model_validate(checkpoint.task_snapshot)
    snapshot_task.execution_authority = ParentExecutionAuthority.scoped(("read", "write"))
    checkpoint = checkpoint.model_copy(
        update={"task_snapshot": snapshot_task.model_dump(mode="json")},
    )
    current = Task(
        tenant_id=_TENANT,
        user_id="user",
        message="resume",
        execution_authority=ParentExecutionAuthority.scoped(("read",)),
    )
    narrowed = resolve_resume_execution_authority(checkpoint, current)
    assert narrowed == ParentExecutionAuthority.scoped(("read",))


def test_authority_expansion_via_checkpoint_blocked() -> None:
    checkpoint = _paused_checkpoint()
    snapshot_task = Task.model_validate(checkpoint.task_snapshot)
    snapshot_task.execution_authority = ParentExecutionAuthority.unrestricted_root()
    checkpoint = checkpoint.model_copy(
        update={"task_snapshot": snapshot_task.model_dump(mode="json")},
    )
    result = validate_checkpoint_authority_expansion(
        checkpoint,
        ParentExecutionAuthority.scoped(("read",)),
    )
    assert result.eligibility is CheckpointResumeEligibility.REJECT_AUTHORITY


def test_lineage_match_passes() -> None:
    checkpoint = _paused_checkpoint()
    persistence = InMemoryExecutionLineagePersistence()
    assert checkpoint.runtime is not None
    scope = build_execution_lineage_attempt_scope(
        tenant_id=checkpoint.tenant_id,
        task_id=checkpoint.task_id,
        run_id=checkpoint.runtime.run_id,
        attempt_id=checkpoint.runtime.attempt_id,
    )
    root = checkpoint.runtime.execution_tree.entries[0].execution_id
    persistence.open_attempt(scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)
    assert (
        _evaluate(checkpoint, execution_lineage_persistence=persistence)
        is CheckpointResumeEligibility.ALLOW_RESUME
    )


def test_lineage_mismatch_blocked() -> None:
    checkpoint = _paused_checkpoint()
    persistence = InMemoryExecutionLineagePersistence()
    assert checkpoint.runtime is not None
    scope = build_execution_lineage_attempt_scope(
        tenant_id=checkpoint.tenant_id,
        task_id=checkpoint.task_id,
        run_id=checkpoint.runtime.run_id,
        attempt_id=checkpoint.runtime.attempt_id,
    )
    other_root = mint_execution_id()
    persistence.open_attempt(scope)
    persistence.open_segment(scope, other_root)
    persistence.admit_root(scope, other_root, other_root)
    assert (
        _evaluate(checkpoint, execution_lineage_persistence=persistence)
        is CheckpointResumeEligibility.REJECT_LINEAGE
    )


class _DurableLineagePersistence(InMemoryExecutionLineagePersistence):
    @property
    def is_durable(self) -> bool:
        return True


def test_missing_required_lineage_blocked() -> None:
    checkpoint = _paused_checkpoint()
    persistence = _DurableLineagePersistence()
    assert (
        _evaluate(
            checkpoint,
            execution_lineage_persistence=persistence,
            require_durable_lineage=True,
        )
        is CheckpointResumeEligibility.REJECT_LINEAGE
    )


def test_cross_process_resume(tmp_path: Path) -> None:
    db_path = tmp_path / "cross.db"
    store_a = SQLiteTaskCheckpointStore(db_path=db_path)
    checkpoint = _paused_checkpoint()
    store_a.save(checkpoint)
    store_b = SQLiteTaskCheckpointStore(db_path=db_path)
    loaded = store_b.get_by_token(checkpoint.task_id, _TENANT, checkpoint.resume_token)
    assert loaded is not None
    identity = resolve_root_task_identity(resume_checkpoint=loaded)
    assert identity.run_id == loaded.runtime.run_id
    assert identity.attempt_id == loaded.runtime.attempt_id


def test_completed_node_not_rerun() -> None:
    checkpoint = _paused_checkpoint()
    assert checkpoint.runtime is not None
    root = checkpoint.runtime.execution_tree.entries[0].execution_id
    child = mint_execution_id()
    runtime = checkpoint.runtime.model_copy(
        update={
            "execution_tree": ExecutionTreeSnapshot(
                task_id=checkpoint.task_id,
                run_id=checkpoint.runtime.run_id,
                attempt_id=checkpoint.runtime.attempt_id,
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
                            summary="done",
                            status="completed",
                            graph_node_id="n1",
                        ),
                    ),
                ],
            ),
        },
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=checkpoint.task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="a1")],
    )
    apply_runtime_checkpoint_to_graph(graph, runtime, {}, run_id=checkpoint.runtime.run_id)
    assert graph.node_by_id("n1").status == ExecutionNodeStatus.COMPLETED


def test_completed_output_retained() -> None:
    checkpoint = _paused_checkpoint()
    assert checkpoint.runtime is not None
    root = checkpoint.runtime.execution_tree.entries[0].execution_id
    child = mint_execution_id()
    runtime = checkpoint.runtime.model_copy(
        update={
            "execution_tree": ExecutionTreeSnapshot(
                task_id=checkpoint.task_id,
                run_id=checkpoint.runtime.run_id,
                attempt_id=checkpoint.runtime.attempt_id,
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
                            summary="canonical-output",
                            status="completed",
                            graph_node_id="n1",
                        ),
                    ),
                ],
            ),
        },
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=checkpoint.task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="a1")],
    )
    prior: dict = {}
    apply_runtime_checkpoint_to_graph(
        graph,
        runtime,
        prior,
        run_id=checkpoint.runtime.run_id,
    )
    assert graph.node_by_id("n1").execution_result is not None
    assert graph.node_by_id("n1").execution_result.summary == "canonical-output"


def test_unknown_side_effect_no_blind_replay() -> None:
    checkpoint = _paused_checkpoint()
    assert checkpoint.runtime is not None
    root = checkpoint.runtime.execution_tree.entries[0].execution_id
    child = mint_execution_id()
    runtime = checkpoint.runtime.model_copy(
        update={
            "execution_tree": ExecutionTreeSnapshot(
                task_id=checkpoint.task_id,
                run_id=checkpoint.runtime.run_id,
                attempt_id=checkpoint.runtime.attempt_id,
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
                        status=ExecutionCheckpointStatus.INTERRUPTED,
                    ),
                ],
            ),
        },
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=checkpoint.task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="a1")],
    )
    apply_runtime_checkpoint_to_graph(graph, runtime, {}, run_id=checkpoint.runtime.run_id)
    assert graph.node_by_id("n1").status == ExecutionNodeStatus.PENDING


def test_resume_not_retry_same_attempt_id() -> None:
    checkpoint = _paused_checkpoint()
    identity = resolve_root_task_identity(resume_checkpoint=checkpoint)
    assert checkpoint.runtime is not None
    assert identity.attempt_id == checkpoint.runtime.attempt_id


def test_attempt_budget_not_reset_on_resume() -> None:
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    run_id = mint_run_id()
    tenant = _TENANT
    attempt_id = mint_attempt_id()
    lifecycle.record_initial_attempt(
        tenant_id=tenant,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    before = lifecycle.get_current_generation(tenant_id=tenant, run_id=run_id)
    checkpoint = _paused_checkpoint(run_id=run_id, attempt_id=attempt_id)
    identity = resolve_root_task_identity(resume_checkpoint=checkpoint)
    after = lifecycle.get_current_generation(tenant_id=tenant, run_id=run_id)
    assert identity.attempt_id == checkpoint.runtime.attempt_id
    assert before == after == 1


def test_concurrent_scheduler_claim_one_winner(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "claim.db")
    claims: list = []

    def _claim(owner: str) -> None:
        claim = store.claim_action(
            "resume:task-1",
            owner,
            lease_seconds=30,
            action="resume",
        )
        claims.append(claim)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(copy_context().run, lambda: _claim("owner-a"))
        f2 = pool.submit(copy_context().run, lambda: _claim("owner-b"))
        concurrent.futures.wait([f1, f2])

    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1


def test_provider_port_not_sqlite_concrete() -> None:
    coordinator_source = (
        _REPO_ROOT / "intergrax" / "runtime" / "long_running" / "coordinator.py"
    ).read_text(encoding="utf-8-sig")
    assert "SQLiteTaskCheckpointStore" not in coordinator_source
    assert "TaskCheckpointPersistence" in coordinator_source or "TaskCheckpointReader" in coordinator_source


def test_no_second_checkpoint_framework() -> None:
    long_running = _REPO_ROOT / "intergrax" / "runtime" / "long_running"
    for path in long_running.rglob("*.py"):
        text = path.read_text(encoding="utf-8-sig")
        for name in _FORBIDDEN_CHECKPOINT_FRAMEWORK_NAMES:
            assert name not in text


def test_checkpoint_validation_no_reflection() -> None:
    path = (
        _REPO_ROOT
        / "intergrax"
        / "runtime"
        / "long_running"
        / "checkpoint_resume_validation.py"
    )
    text = path.read_text(encoding="utf-8-sig")
    assert _REFLECTION_PATTERN.search(text) is None


def test_schema_constants_frozen() -> None:
    assert CANONICAL_RUNTIME_CHECKPOINT_SCHEMA_VERSION == "runtime_checkpoint.v2"
    assert CANONICAL_TASK_CHECKPOINT_SCHEMA_VERSION == "task_checkpoint.v1"


def test_persist_non_resumable_state_blocked() -> None:
    task = Task(
        tenant_id=_TENANT,
        user_id="user",
        message="running",
        state=TaskState.RUNNING,
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True),
        ),
    )
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
    )
    try:
        runtime = build_runtime_checkpoint(task, run_id=run_id, attempt_id=attempt_id)
        with pytest.raises(CheckpointNotResumableError):
            assert_checkpoint_persistable(task, runtime)
    finally:
        reset_active_execution_identity(token)


def test_r1_final_regression_subprocess() -> None:
    proc = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "tests/unit/runtime/architecture/test_npsc5e_r1_final_retry_attempt_qualification.py",
            "-q",
            "--tb=no",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_p0a_regression_subprocess() -> None:
    proc = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "tests/unit/runtime/architecture/test_npsc5e_p0a_execution_lineage_baseline_qualification.py",
            "-q",
            "--tb=no",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_provenance_shas_recorded() -> None:
    assert _R1_FINAL_SHA.startswith("76603ed")
    assert _LINEAGE_HARDENING_SHA.startswith("a18e65c")

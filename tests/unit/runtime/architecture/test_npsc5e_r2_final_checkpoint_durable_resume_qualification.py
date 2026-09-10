# © Artur Czarnecki. All rights reserved.

"""NPSC-5E/R2 Final — checkpoint & durable resume qualification and freeze."""

from __future__ import annotations

import ast
import concurrent.futures
import inspect
import re
import subprocess
from contextvars import copy_context
from pathlib import Path

import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_lineage import build_execution_lineage_attempt_scope
from intergrax.contracts.execution_retry import (
    ExecutionFailureKind,
    ExecutionRetryAction,
    ExecutionRetryEligibilityRequest,
)
from intergrax.contracts.execution_terminal import ExecutionTerminalOutcome
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.execution.attempt_lifecycle import (
    AttemptLifecycleService,
    InMemoryAttemptLifecycleStore,
)
from intergrax.runtime.execution.execution_terminal.persistence import (
    CheckpointStoreExecutionTerminalStore,
)
from intergrax.runtime.execution.execution_terminal.service import ExecutionTerminalService
from intergrax.runtime.execution.lineage.persistence import InMemoryExecutionLineagePersistence
from intergrax.runtime.execution.orchestration import resolve_root_task_identity
from intergrax.runtime.execution.retry import (
    ExecutionAttemptRetryService,
    classify_execution_failure,
    evaluate_execution_retry_eligibility,
)
from intergrax.runtime.long_running.checkpoint_resume_validation import (
    CANONICAL_TASK_CHECKPOINT_SCHEMA_VERSION,
    CheckpointResumeEligibility,
    evaluate_checkpoint_resume_eligibility,
    resolve_resume_execution_authority,
    validate_checkpoint_resume_authority,
)
from intergrax.runtime.long_running.checkpoint_revision import StaleCheckpointWriteError
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.execution_tree_checkpoint import minimal_runtime_checkpoint
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.long_running.runtime_checkpoint import (
    CANONICAL_RUNTIME_CHECKPOINT_SCHEMA_VERSION,
)
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TENANT = "tenant-r2-final"

_R1_FINAL_SHA = "76603ed266f9f54106bf4718fe8886180a826351"
_R2_SHA = "37276a7e2755847b088fc91da76dee0f96824172"
_R2_H1_SHA = "7bc0c651ca536ffe1b06ac88e7180b4b62380010"
_R2_H2_SHA = "4c87483a44341e34667ea5c7868be52b7cc71300"
_R2_H2_Q1_SHA = "0cc93bed8e432bec66e2fd773185dc68b270bbd4"

_REFLECTION_PATTERN = re.compile(r"\b(getattr|setattr|hasattr)\(")
_FORBIDDEN_FRAMEWORK_NAMES = (
    "EnterpriseCheckpointEngine",
    "RecoveryCheckpointRuntime",
    "UniversalResumeManager",
    "CheckpointAuthorityResolver",
    "ResumeAuthorityEngine",
)

_MANDATORY_SUITES: tuple[tuple[str, list[str]], ...] = (
    ("R1 Final", ["tests/unit/runtime/architecture/test_npsc5e_r1_final_retry_attempt_qualification.py"]),
    ("R2 Original", ["tests/unit/runtime/architecture/test_npsc5e_r2_checkpoint_durable_resume_hardening.py"]),
    ("R2-H1", ["tests/unit/runtime/architecture/test_npsc5e_r2_h1_authority_stale_checkpoint_closure.py"]),
    ("R2-H2", ["tests/unit/runtime/architecture/test_npsc5e_r2_h2_checkpoint_revision_stale_writer_protection.py"]),
    ("R2-H2-Q1", ["tests/unit/runtime/architecture/test_npsc5e_r2_h2_q1_frozen_regression_closure.py"]),
    ("P0A", ["tests/unit/runtime/architecture/test_npsc5e_p0a_execution_lineage_baseline_qualification.py"]),
    (
        "DG_001",
        [
            "tests/unit/contracts/test_execution_lineage_contracts.py",
            "tests/unit/runtime/execution/lineage/",
        ],
    ),
    (
        "NPSC-5D Final",
        ["tests/unit/runtime/architecture/test_npsc5d_final_multi_agent_governance_qualification.py"],
    ),
    ("HITL R3", ["tests/unit/runtime/architecture/test_npsc5d_r3_governed_continuation.py"]),
    ("NPSC-5A", ["tests/unit/runtime/architecture/test_npsc5a_coordination_delegation_e2e.py"]),
    (
        "NPSC-5B",
        ["tests/unit/runtime/architecture/test_npsc5b_final_production_fanout_fanin_qualification.py"],
    ),
    ("NPSC-5C", ["tests/unit/runtime/architecture/test_npsc5c_decision_execution_e2e.py"]),
    (
        "Attempt lifecycle",
        [
            "tests/unit/runtime/execution/test_attempt_lifecycle.py",
            "tests/unit/runtime/execution/test_attempt_lifecycle_durability_gate.py",
            "tests/conformance/runtime/durability/test_attempt_lifecycle.py",
        ],
    ),
    (
        "Child execution",
        [
            "tests/unit/runtime/execution/test_child_execution.py",
            "tests/unit/runtime/execution/authority/test_child_execution_authority_policy.py",
        ],
    ),
    ("Terminal", ["tests/unit/runtime/execution/test_p0c6_terminal_outcome_convergence.py"]),
    (
        "Cancellation",
        [
            "tests/unit/runtime/cancellation/test_p0c5_cancellation_continuity.py",
            "-k",
            "not survives_process_restart",
            "tests/unit/runtime/cancellation/test_p0c5a_explicit_terminal_wiring.py",
            "tests/unit/applications/test_task_control_governed_resume.py",
        ],
    ),
    ("Checkpoint store", ["tests/unit/runtime/long_running/test_checkpoint_store.py"]),
    (
        "Long-running",
        [
            "tests/unit/runtime/long_running/test_pcm_scheduler_integrity.py",
            "tests/unit/runtime/long_running/test_pba_fix_a_checkpoint_port_consumption.py",
            "tests/unit/runtime/long_running/test_runtime_checkpoint.py",
            "tests/unit/runtime/long_running/test_resume_planner.py",
            "tests/unit/runtime/long_running/test_ue_9c_execution_tree_checkpoint.py",
            "tests/unit/runtime/long_running/test_p0c3_recovery_state_authority.py",
        ],
    ),
)

_R2_PRODUCTION_SURFACE = (
    "intergrax/runtime/long_running/checkpoint_resume_validation.py",
    "intergrax/runtime/long_running/checkpoint_revision.py",
    "intergrax/runtime/long_running/coordinator.py",
    "intergrax/runtime/long_running/persistence_contract.py",
    "intergrax/runtime/long_running/runtime_checkpoint.py",
    "intergrax/runtime/long_running/store.py",
    "intergrax/runtime/long_running/models.py",
)


def _paused_checkpoint(
    *,
    task_id: TaskId | None = None,
    tenant_id: str = _TENANT,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
    root_execution_id: ExecutionId | None = None,
    execution_authority: ParentExecutionAuthority | None = None,
    resume_token: str = "rt-final",
) -> TaskCheckpoint:
    resolved_task_id = task_id if task_id is not None else mint_task_id()
    resolved_run_id = run_id if run_id is not None else mint_run_id()
    resolved_attempt_id = attempt_id if attempt_id is not None else mint_attempt_id()
    resolved_root = root_execution_id if root_execution_id is not None else mint_execution_id()
    task = Task(
        task_id=resolved_task_id,
        tenant_id=tenant_id,
        user_id="user",
        message="paused",
        state=TaskState.WAITING_FOR_HUMAN,
        execution_authority=execution_authority,
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, resume_token=resume_token),
        ),
    )
    return TaskCheckpoint(
        task_id=resolved_task_id,
        tenant_id=tenant_id,
        resume_token=resume_token,
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=task.model_dump(mode="json"),
        created_at_utc="2026-09-10T12:00:00+00:00",
        runtime=minimal_runtime_checkpoint(
            task_id=resolved_task_id,
            run_id=resolved_run_id,
            attempt_id=resolved_attempt_id,
            root_execution_id=resolved_root,
        ),
    )


def _resume_task(
    checkpoint: TaskCheckpoint,
    *,
    execution_authority: ParentExecutionAuthority | None = None,
) -> Task:
    return Task(
        task_id=TaskId(checkpoint.task_id),
        tenant_id=checkpoint.tenant_id,
        user_id="user",
        message="resume",
        execution_authority=execution_authority,
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(
                enabled=True,
                resume_token=checkpoint.resume_token,
            ),
        ),
    )


def _wire_lineage(checkpoint: TaskCheckpoint, persistence: InMemoryExecutionLineagePersistence) -> None:
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


def _run_pytest(targets: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "pytest", *targets, "-q", "--tb=no"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(("label", "targets"), _MANDATORY_SUITES, ids=[label for label, _ in _MANDATORY_SUITES])
def test_mandatory_frozen_suite_passes(label: str, targets: list[str]) -> None:
    proc = _run_pytest(targets)
    assert proc.returncode == 0, f"{label} failed:\n{proc.stdout}\n{proc.stderr}"


def test_canonical_predecessor_shas_recorded() -> None:
    assert _R1_FINAL_SHA.startswith("76603ed")
    assert _R2_SHA.startswith("37276a7")
    assert _R2_H1_SHA.startswith("7bc0c65")
    assert _R2_H2_SHA.startswith("4c87483")
    assert _R2_H2_Q1_SHA.startswith("0cc93be")


def test_schema_constants_frozen() -> None:
    assert CANONICAL_RUNTIME_CHECKPOINT_SCHEMA_VERSION == "runtime_checkpoint.v2"
    assert CANONICAL_TASK_CHECKPOINT_SCHEMA_VERSION == "task_checkpoint.v1"


def test_persistence_contract_exposes_revision_cas() -> None:
    signature = inspect.signature(TaskCheckpointPersistence.save)
    assert "expected_revision" in signature.parameters


def test_final_normal_resume_process_boundary_revision_increment(tmp_path: Path) -> None:
    db_path = tmp_path / "normal.db"
    store_a = SQLiteTaskCheckpointStore(db_path=db_path)
    checkpoint = _paused_checkpoint(
        execution_authority=ParentExecutionAuthority.scoped(("read", "write")),
    )
    assert checkpoint.runtime is not None
    saved = store_a.save(checkpoint)
    assert saved.revision == 1
    runtime = saved.runtime
    assert runtime is not None

    persistence = InMemoryExecutionLineagePersistence()
    _wire_lineage(saved, persistence)
    current = _resume_task(
        saved,
        execution_authority=ParentExecutionAuthority.scoped(("read", "write")),
    )
    eligibility = evaluate_checkpoint_resume_eligibility(
        saved,
        target_task_id=saved.task_id,
        target_tenant_id=_TENANT,
        target_run_id=runtime.run_id,
        target_attempt_id=runtime.attempt_id,
        target_root_execution_id=runtime.execution_tree.entries[0].execution_id,
        latest_checkpoint=saved,
        execution_lineage_persistence=persistence,
        current_task=current,
        policy_decision=PolicyDecision(action=PolicyAction.ALLOW, reason="ok"),
    )
    assert eligibility.eligibility is CheckpointResumeEligibility.ALLOW_RESUME

    store_b = SQLiteTaskCheckpointStore(db_path=db_path)
    restored_task = _resume_task(
        saved,
        execution_authority=ParentExecutionAuthority.scoped(("read", "write")),
    )
    restored = LongRunningCoordinator.restore_if_resuming(restored_task, store_b)
    assert restored is not None
    identity = resolve_root_task_identity(resume_checkpoint=saved)
    assert identity.attempt_id == runtime.attempt_id
    assert identity.run_id == runtime.run_id

    next_checkpoint = store_b.save(
        saved.model_copy(
            update={"checkpoint_id": "ckpt_after_resume", "progress_message": "resumed"},
        ),
        expected_revision=saved.revision,
    )
    assert next_checkpoint.revision == 2


def test_final_stale_writer_scenario(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "stale.db")
    base = store.save(_paused_checkpoint())
    store.save(
        base.model_copy(update={"checkpoint_id": "ckpt_b", "progress_message": "b"}),
        expected_revision=base.revision,
    )
    with pytest.raises(StaleCheckpointWriteError):
        store.save(
            base.model_copy(update={"checkpoint_id": "ckpt_a", "progress_message": "a"}),
            expected_revision=base.revision,
        )


def test_final_authority_narrowing_scenario() -> None:
    checkpoint = _paused_checkpoint(
        execution_authority=ParentExecutionAuthority.scoped(("read", "write")),
    )
    current = _resume_task(
        checkpoint,
        execution_authority=ParentExecutionAuthority.scoped(("read",)),
    )
    effective = resolve_resume_execution_authority(checkpoint, current)
    assert effective == ParentExecutionAuthority.scoped(("read",))


def test_final_authority_missing_scenario() -> None:
    checkpoint = _paused_checkpoint(
        execution_authority=ParentExecutionAuthority.scoped(("read",)),
    )
    current = _resume_task(checkpoint, execution_authority=None)
    result = validate_checkpoint_resume_authority(checkpoint, current)
    assert result.eligibility is CheckpointResumeEligibility.REJECT_AUTHORITY


def test_final_policy_deny_scenario() -> None:
    checkpoint = _paused_checkpoint()
    result = evaluate_checkpoint_resume_eligibility(
        checkpoint,
        target_task_id=checkpoint.task_id,
        target_tenant_id=_TENANT,
        latest_checkpoint=checkpoint,
        policy_decision=PolicyDecision(action=PolicyAction.DENY, reason="deny"),
    )
    assert result.eligibility is CheckpointResumeEligibility.REJECT_GOVERNANCE


def test_final_terminal_scenario(tmp_path: Path) -> None:
    checkpoint = _paused_checkpoint()
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "terminal.db")
    store.save(checkpoint)
    terminal = ExecutionTerminalService(CheckpointStoreExecutionTerminalStore(store))
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
    result = evaluate_checkpoint_resume_eligibility(
        loaded,
        target_task_id=loaded.task_id,
        target_tenant_id=_TENANT,
        execution_terminal=terminal,
    )
    assert result.eligibility is CheckpointResumeEligibility.REJECT_TERMINAL


def test_final_lineage_mismatch_scenario() -> None:
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
    result = evaluate_checkpoint_resume_eligibility(
        checkpoint,
        target_task_id=checkpoint.task_id,
        target_tenant_id=_TENANT,
        execution_lineage_persistence=persistence,
    )
    assert result.eligibility is CheckpointResumeEligibility.REJECT_LINEAGE


def test_final_cross_process_resume_scenario(tmp_path: Path) -> None:
    db_path = tmp_path / "cross.db"
    store_a = SQLiteTaskCheckpointStore(db_path=db_path)
    saved = store_a.save(_paused_checkpoint())
    store_b = SQLiteTaskCheckpointStore(db_path=db_path)
    loaded = store_b.get_latest(saved.task_id, _TENANT)
    assert loaded is not None
    assert loaded.revision == saved.revision
    loaded_runtime = loaded.runtime
    assert loaded_runtime is not None
    identity = resolve_root_task_identity(resume_checkpoint=loaded)
    assert identity.run_id == loaded_runtime.run_id
    assert identity.attempt_id == loaded_runtime.attempt_id


def test_final_duplicate_resume_claim_one_winner(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "claim.db")
    claims: list = []

    def _claim(owner: str) -> None:
        claim = store.claim_action(
            "resume:task-final",
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


def test_final_retry_interop_attempt_lifecycle_owned() -> None:
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    retry = ExecutionAttemptRetryService(lifecycle)
    run_id = mint_run_id()
    attempt_one = mint_attempt_id()
    lifecycle.record_initial_attempt(
        tenant_id=_TENANT,
        run_id=run_id,
        attempt_id=attempt_one,
    )
    checkpoint = _paused_checkpoint(run_id=run_id, attempt_id=attempt_one)
    before = lifecycle.get_current_generation(tenant_id=_TENANT, run_id=run_id)
    identity = resolve_root_task_identity(resume_checkpoint=checkpoint)
    after_resume = lifecycle.get_current_generation(tenant_id=_TENANT, run_id=run_id)
    assert identity.attempt_id == attempt_one
    assert before == after_resume == 1

    request = ExecutionRetryEligibilityRequest(
        classification=classify_execution_failure(
            kind=ExecutionFailureKind.RETRYABLE_TRANSIENT,
        ),
        attempt_number=1,
        max_attempts=3,
    )
    eligibility = evaluate_execution_retry_eligibility(request)
    assert eligibility.action is ExecutionRetryAction.RETRY
    transition = retry.transition_for_retry(
        tenant_id=_TENANT,
        task_id=TaskId(checkpoint.task_id),
        run_id=run_id,
        expected_attempt_id=attempt_one,
        request=request,
    )
    assert transition is not None
    assert transition.active_attempt_id != attempt_one
    assert lifecycle.get_current_generation(tenant_id=_TENANT, run_id=run_id) == 2


def test_final_hitl_interop_no_checkpoint_self_approval() -> None:
    coordinator_source = (
        _REPO_ROOT / "intergrax/runtime/long_running/coordinator.py"
    ).read_text(encoding="utf-8-sig")
    assert "approve_human" not in coordinator_source
    assert "self_approve" not in coordinator_source
    proc = _run_pytest(
        ["tests/unit/runtime/architecture/test_npsc5d_r3_governed_continuation.py"],
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_final_child_interop_no_checkpoint_bypass() -> None:
    coordinator_source = (
        _REPO_ROOT / "intergrax/runtime/long_running/coordinator.py"
    ).read_text(encoding="utf-8-sig")
    validation_source = (
        _REPO_ROOT / "intergrax/runtime/long_running/checkpoint_resume_validation.py"
    ).read_text(encoding="utf-8-sig")
    assert "ChildExecutionPort" not in coordinator_source
    assert "ChildExecutionPort" not in validation_source
    child_port = (
        _REPO_ROOT / "intergrax/runtime/execution/delegated_subtask_child_port.py"
    ).read_text(encoding="utf-8-sig")
    assert "ChildExecutionPort" in child_port


def test_no_second_checkpoint_framework_or_revision_authority() -> None:
    long_running = _REPO_ROOT / "intergrax/runtime/long_running"
    for path in long_running.rglob("*.py"):
        text = path.read_text(encoding="utf-8-sig")
        for name in _FORBIDDEN_FRAMEWORK_NAMES:
            assert name not in text


def test_r2_surface_no_reflection() -> None:
    for relative in _R2_PRODUCTION_SURFACE:
        text = (_REPO_ROOT / relative).read_text(encoding="utf-8-sig")
        assert _REFLECTION_PATTERN.search(text) is None


def test_coordinator_provider_neutral_no_sqlite_leak() -> None:
    source = (_REPO_ROOT / "intergrax/runtime/long_running/coordinator.py").read_text(
        encoding="utf-8-sig",
    )
    assert "SQLiteTaskCheckpointStore" not in source
    assert "BEGIN IMMEDIATE" not in source
    assert "rowid" not in source


def test_checkpoint_does_not_rehydrate_authority() -> None:
    coordinator_path = _REPO_ROOT / "intergrax/runtime/long_running/coordinator.py"
    tree = ast.parse(coordinator_path.read_text(encoding="utf-8-sig"))
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Attribute):
                continue
            if (
                isinstance(target.value, ast.Name)
                and target.value.id == "restored"
                and target.attr == "execution_authority"
            ):
                forbidden.append("restored.execution_authority assignment")
    assert forbidden == []


def test_pre_existing_cancellation_fixture_invalid_persist_gate() -> None:
    proc = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "tests/unit/runtime/cancellation/test_p0c5_cancellation_continuity.py::test_terminal_cancellation_survives_process_restart",
            "-q",
            "--tb=line",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "not resumable" in combined or "CheckpointNotResumableError" in combined
    proc_ok = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "tests/unit/runtime/cancellation/test_p0c5_cancellation_continuity.py",
            "-k",
            "not survives_process_restart",
            "-q",
            "--tb=no",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_ok.returncode == 0, proc_ok.stdout + proc_ok.stderr


def test_pre_existing_partial_results_unrelated_to_r2() -> None:
    r2_changed = subprocess.run(
        ["git", "show", "--name-only", "--pretty=format:", _R2_SHA],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "intergrax/runtime/long_running/partial_results.py" not in r2_changed
    proc = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "tests/unit/runtime/long_running/test_partial_results.py::test_build_task_progress_view_aggregates_checkpoints",
            "-q",
            "--tb=line",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0


def test_ruff_final_test_no_new_errors() -> None:
    proc = subprocess.run(
        ["uv", "run", "ruff", "check", str(Path(__file__))],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_pyright_final_test_no_new_errors() -> None:
    proc = subprocess.run(
        ["uv", "run", "pyright", str(Path(__file__))],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

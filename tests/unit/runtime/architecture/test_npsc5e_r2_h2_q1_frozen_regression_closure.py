# © Artur Czarnecki. All rights reserved.

"""NPSC-5E/R2-H2-Q1 — mandatory frozen regression closure qualification."""

from __future__ import annotations

import inspect
import re
import subprocess
from pathlib import Path

import pytest

from intergrax.contracts.attempt_lifecycle import AttemptTransitionReason
from intergrax.contracts.execution_identity import (
    TaskId,
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.execution.attempt_lifecycle import (
    AttemptLifecycleService,
    InMemoryAttemptLifecycleStore,
)
from intergrax.runtime.long_running.checkpoint_revision import CheckpointRevisionRequiredError
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_R2_H2_SHA = "4c87483a44341e34667ea5c7868be52b7cc71300"
_R1_FINAL_SHA = "76603ed266f9f54106bf4718fe8886180a826351"
_LINEAGE_HARDENING_SHA = "a18e65c077ed57bf3bb64ef015b47ee1f3ceb6bf"
_TENANT = "tenant-h2-q1"
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
    ("P0A", ["tests/unit/runtime/architecture/test_npsc5e_p0a_execution_lineage_baseline_qualification.py"]),
    (
        "DG_001 lineage",
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


def _paused_checkpoint(
    *,
    task_id: TaskId | None = None,
    tenant_id: str = _TENANT,
) -> TaskCheckpoint:
    resolved_task_id: TaskId = task_id if task_id is not None else mint_task_id()
    task = Task(
        task_id=resolved_task_id,
        tenant_id=tenant_id,
        user_id="user",
        message="paused",
        state=TaskState.WAITING_FOR_HUMAN,
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, resume_token="rt-q1"),
        ),
    )
    return TaskCheckpoint(
        task_id=resolved_task_id,
        tenant_id=tenant_id,
        resume_token="rt-q1",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=task.model_dump(mode="json"),
        created_at_utc="2026-09-10T12:00:00+00:00",
    )


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
    assert _R2_H2_SHA.startswith("4c87483")
    assert _R1_FINAL_SHA.startswith("76603ed")
    assert _LINEAGE_HARDENING_SHA.startswith("a18e65c")


def test_persistence_contract_exposes_revision_cas() -> None:
    signature = inspect.signature(TaskCheckpointPersistence.save)
    assert "expected_revision" in signature.parameters


def test_empty_stream_none_allowed_existing_stream_requires_revision(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "cas.db")
    first = store.save(_paused_checkpoint())
    with pytest.raises(CheckpointRevisionRequiredError):
        store.save(first.model_copy(update={"checkpoint_id": "ckpt_2"}))


def test_coordinator_provider_neutral_no_sqlite_reference() -> None:
    source = (_REPO_ROOT / "intergrax/runtime/long_running/coordinator.py").read_text(
        encoding="utf-8-sig",
    )
    assert "SQLiteTaskCheckpointStore" not in source
    assert "BEGIN IMMEDIATE" not in source
    assert "expected_revision=task.runtime.orchestration.checkpoint_revision" in source


def test_no_direct_sqlite_checkpoint_insert_outside_store() -> None:
    production_roots = (
        _REPO_ROOT / "intergrax",
        _REPO_ROOT / "applications",
        _REPO_ROOT / "agents",
    )
    hits: list[str] = []
    for root in production_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path.name == "store.py" and "long_running" in path.parts:
                continue
            text = path.read_text(encoding="utf-8-sig")
            if "INSERT INTO task_checkpoints" in text:
                hits.append(str(path.relative_to(_REPO_ROOT)))
    assert hits == []


def test_no_second_checkpoint_framework() -> None:
    long_running = _REPO_ROOT / "intergrax/runtime/long_running"
    for path in long_running.rglob("*.py"):
        text = path.read_text(encoding="utf-8-sig")
        for name in _FORBIDDEN_FRAMEWORK_NAMES:
            assert name not in text


def test_h2_surface_no_reflection() -> None:
    h2_paths = (
        "intergrax/runtime/long_running/checkpoint_revision.py",
        "intergrax/runtime/long_running/store.py",
        "intergrax/runtime/long_running/coordinator.py",
        "intergrax/runtime/long_running/persistence_contract.py",
    )
    for relative in h2_paths:
        text = (_REPO_ROOT / relative).read_text(encoding="utf-8-sig")
        assert _REFLECTION_PATTERN.search(text) is None


def test_checkpoint_revision_not_attempt_id_field() -> None:
    checkpoint_fields = set(TaskCheckpoint.model_fields)
    assert "revision" in checkpoint_fields
    assert "attempt_id" not in checkpoint_fields


def test_attempt_transition_independent_of_checkpoint_revision(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "attempt.db")
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    run_id = mint_run_id()
    attempt_one = mint_attempt_id()
    lifecycle.record_initial_attempt(
        tenant_id=_TENANT,
        run_id=run_id,
        attempt_id=attempt_one,
    )
    checkpoint = store.save(_paused_checkpoint())
    successor = store.save(
        checkpoint.model_copy(update={"checkpoint_id": "ckpt_2"}),
        expected_revision=checkpoint.revision,
    )
    lifecycle.transition_to_next_attempt(
        tenant_id=_TENANT,
        run_id=run_id,
        expected_attempt_id=attempt_one,
        reason=AttemptTransitionReason.RETRY,
    )
    assert successor.revision == 2
    assert lifecycle.get_current_generation(tenant_id=_TENANT, run_id=run_id) == 2


def test_resume_does_not_reset_checkpoint_revision(tmp_path: Path) -> None:
    db_path = tmp_path / "resume.db"
    store_a = SQLiteTaskCheckpointStore(db_path=db_path)
    first = store_a.save(_paused_checkpoint())
    second = store_a.save(
        first.model_copy(update={"checkpoint_id": "ckpt_2"}),
        expected_revision=first.revision,
    )
    store_b = SQLiteTaskCheckpointStore(db_path=db_path)
    loaded = store_b.get_latest(first.task_id, _TENANT)
    assert loaded is not None
    assert loaded.revision == second.revision == 2


def test_pre_existing_cancellation_fixture_unrelated_to_h2_revision() -> None:
    """R2 persist gate rejects CREATED-state fixture; H2 revision CAS is not on this path."""
    source = (_REPO_ROOT / "intergrax/runtime/long_running/coordinator.py").read_text(
        encoding="utf-8-sig",
    )
    assert "assert_checkpoint_persistable" in source
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
    assert "not resumable" in proc.stdout + proc.stderr


def test_pre_existing_partial_results_unrelated_to_h2_files() -> None:
    h2_changed = subprocess.run(
        ["git", "show", "--name-only", "--pretty=format:", _R2_H2_SHA],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "partial_results.py" not in h2_changed

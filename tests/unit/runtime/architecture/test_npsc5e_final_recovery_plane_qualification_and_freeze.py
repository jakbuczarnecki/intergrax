# © Artur Czarnecki. All rights reserved.

"""NPSC-5E Final — recovery plane (R1+R2+R3) qualification and freeze."""

from __future__ import annotations

import ast
import inspect
import re
import subprocess
from pathlib import Path

import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    TaskId,
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_retry import (
    ExecutionFailureKind,
    ExecutionRetryAction,
    ExecutionRetryEligibilityRequest,
)
from intergrax.contracts.execution_terminal import ExecutionTerminalOutcome
from intergrax.contracts.partial_recovery import (
    SlotRecoveryDisposition,
    SlotRecoveryPolicyAction,
    SlotRecoveryPolicyRequest,
    evaluate_slot_recovery_policy,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.execution.attempt_lifecycle import (
    AttemptLifecycleService,
    InMemoryAttemptLifecycleStore,
)
from intergrax.runtime.execution.orchestration import resolve_root_task_identity
from intergrax.runtime.execution.retry import (
    ExecutionAttemptRetryService,
    classify_execution_failure,
    evaluate_execution_retry_eligibility,
)
from intergrax.runtime.long_running.checkpoint_resume_validation import (
    CheckpointResumeEligibility,
    evaluate_checkpoint_resume_eligibility,
)
from intergrax.runtime.long_running.checkpoint_revision import StaleCheckpointWriteError
from intergrax.runtime.execution.execution_terminal.persistence import (
    CheckpointStoreExecutionTerminalStore,
)
from intergrax.runtime.execution.execution_terminal.service import ExecutionTerminalService
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from tests.unit.runtime.architecture.test_npsc5e_r2_final_checkpoint_durable_resume_qualification import (
    _TENANT as _R2_TENANT,
)
from tests.unit.runtime.architecture.test_npsc5e_r2_final_checkpoint_durable_resume_qualification import (
    _paused_checkpoint,
    _resume_task,
    _run_pytest,
)
from tests.unit.runtime.architecture.test_npsc5e_r3_child_fanout_partial_recovery import (
    _FORBIDDEN_RECOVERY_NAMES as _R3_FORBIDDEN_NAMES,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

_R1_FINAL_SHA = "76603ed266f9f54106bf4718fe8886180a826351"
_R2_FINAL_SHA = "c030d1d4752513bc11ec2597de4e760ae551a978"
_R3_FINAL_SHA = "1ed12e21888840e800b632ec66f6b683e5960591"
_NPSC_5D_FINAL_SHA = "a4a1faca01cd5004e372f235132184a84aa5a6bd"

_REFLECTION_PATTERN = re.compile(r"\b(getattr|setattr|hasattr)\(")

_FORBIDDEN_GOD_OBJECTS = (
    "RecoveryEngine",
    "UniversalRecoveryManager",
    "RecoveryRuntime",
    "EnterpriseRecoveryCoordinator",
    "RetryRuntime",
    "ExecutionRecoveryRuntime",
    "EnterpriseRetryRecoveryManager",
    "UnifiedRecoveryEverythingService",
    "EnterpriseCheckpointEngine",
    "RecoveryCheckpointRuntime",
    "UniversalResumeManager",
    "PartialRecoveryRuntime",
    "FanOutRecoveryEngine",
    "ChildRecoveryScheduler",
    "RecoveryOrchestrator",
)

_R1_PRODUCTION_SURFACE = (
    "intergrax/contracts/execution_retry.py",
    "intergrax/runtime/execution/retry/",
)

_R2_PRODUCTION_SURFACE = (
    "intergrax/runtime/long_running/checkpoint_resume_validation.py",
    "intergrax/runtime/long_running/coordinator.py",
    "intergrax/runtime/long_running/runtime_checkpoint.py",
    "intergrax/runtime/long_running/persistence_contract.py",
    "intergrax/runtime/long_running/store.py",
)

_R3_PRODUCTION_SURFACE = (
    "intergrax/contracts/partial_recovery.py",
    "intergrax/runtime/long_running/topology_recovery_snapshot.py",
    "intergrax/runtime/execution/fan_out_partial_recovery.py",
    "intergrax/runtime/long_running/runtime_checkpoint.py",
)

# R3 Final subprocess-composes R1 Final, R2 Final, R3 implementation gate, and section 94 matrix.
_MANDATORY_SUITES: tuple[tuple[str, list[str]], ...] = (
    (
        "NPSC-5E recovery plane (R1+R2+R3 finals + section 94)",
        [
            "tests/unit/runtime/architecture/test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py",
        ],
    ),
)

_OWNERSHIP_CERTIFICATE: tuple[tuple[str, str], ...] = (
    ("Execution lifecycle", "ExecutionRuntime"),
    ("Attempt lifecycle", "AttemptLifecycleService"),
    ("Retry policy", "ExecutionAttemptRetryService"),
    ("Durable checkpoint", "LongRunningCoordinator / RuntimeCheckpoint"),
    ("Checkpoint logical revision", "TaskCheckpointPersistence"),
    ("Partial recovery", "FanOutPartialRecoveryService (R3)"),
    ("Topology scheduling", "Nexus"),
    ("Child execution", "ChildExecutionPort"),
    ("Lineage", "ExecutionLineagePersistence"),
    ("Terminal", "ExecutionTerminalService"),
    ("Authority", "canonical authority plane"),
    ("Governance", "Governance"),
    ("HITL", "canonical HITL continuation"),
    ("Fan-in", "BoundedMultiAgentFanOutService"),
)


@pytest.mark.parametrize(("label", "targets"), _MANDATORY_SUITES, ids=[label for label, _ in _MANDATORY_SUITES])
def test_mandatory_frozen_suite_passes(label: str, targets: list[str]) -> None:
    proc = _run_pytest(targets)
    assert proc.returncode == 0, f"{label} failed:\n{proc.stdout}\n{proc.stderr}"


def test_section_94_regression_labels_composed_in_r3_final_gate() -> None:
    from tests.unit.runtime.architecture import (
        test_npsc5e_r3_final_child_fanout_partial_recovery_qualification as r3_final,
    )

    labels = {label for label, _ in r3_final._MANDATORY_SUITES}
    assert {"R1 Final", "R2 Final"} <= labels
    required = {
        "P0A",
        "DG_001",
        "NPSC-5A",
        "NPSC-5B Final",
        "NPSC-5C",
        "NPSC-5D Final",
        "HITL R3",
        "Attempt lifecycle",
        "Child execution",
        "Terminal",
        "Cancellation",
        "Checkpoint store",
        "Long-running",
        "Fan-out",
    }
    assert required <= labels


def test_canonical_predecessor_shas_recorded() -> None:
    assert _R1_FINAL_SHA.startswith("76603ed")
    assert _R2_FINAL_SHA.startswith("c030d1d")
    assert _R3_FINAL_SHA.startswith("1ed12e2")
    assert _NPSC_5D_FINAL_SHA.startswith("a4a1fac")


def test_recovery_plane_orthogonality_retry_resume_partial_distinct() -> None:
    r1 = evaluate_execution_retry_eligibility(
        ExecutionRetryEligibilityRequest(
            classification=classify_execution_failure(kind=ExecutionFailureKind.RETRYABLE_TRANSIENT),
            attempt_number=1,
            max_attempts=3,
        ),
    )
    r3_policy = evaluate_slot_recovery_policy(
        SlotRecoveryPolicyRequest(
            disposition=SlotRecoveryDisposition.FAILED,
            failure_kind=ExecutionFailureKind.RETRYABLE_TRANSIENT,
        ),
    )
    assert r1.action is ExecutionRetryAction.RETRY
    assert r3_policy.action is SlotRecoveryPolicyAction.RECOVER
    checkpoint = _paused_checkpoint()
    identity = resolve_root_task_identity(resume_checkpoint=checkpoint)
    assert checkpoint.runtime is not None
    assert identity.attempt_id == checkpoint.runtime.attempt_id
    assert r3_policy.action is not r1.action


def test_no_recovery_god_object_in_production() -> None:
    production_root = _REPO_ROOT / "intergrax"
    forbidden = set(_FORBIDDEN_GOD_OBJECTS) | set(_R3_FORBIDDEN_NAMES)
    hits: list[str] = []
    for path in production_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for name in forbidden:
            if name in source:
                hits.append(f"{path.relative_to(_REPO_ROOT)}:{name}")
    assert not hits, f"forbidden recovery god objects: {hits}"


def test_ownership_certificate_frozen() -> None:
    assert len(_OWNERSHIP_CERTIFICATE) >= 14
    owners = {row[1] for row in _OWNERSHIP_CERTIFICATE}
    assert "AttemptLifecycleService" in owners
    assert "ExecutionAttemptRetryService" in owners
    assert "TaskCheckpointPersistence" in owners


def test_composite_e2e_1_retry_then_resume_preserves_attempt_two(tmp_path: Path) -> None:
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    retry = ExecutionAttemptRetryService(lifecycle)
    run_id = mint_run_id()
    attempt_one = mint_attempt_id()
    lifecycle.record_initial_attempt(tenant_id=_R2_TENANT, run_id=run_id, attempt_id=attempt_one)
    request = ExecutionRetryEligibilityRequest(
        classification=classify_execution_failure(kind=ExecutionFailureKind.RETRYABLE_TRANSIENT),
        attempt_number=1,
        max_attempts=3,
    )
    transition = retry.transition_for_retry(
        tenant_id=_R2_TENANT,
        task_id=mint_task_id(),
        run_id=run_id,
        expected_attempt_id=attempt_one,
        request=request,
    )
    assert transition is not None
    attempt_two = transition.active_attempt_id
    assert lifecycle.get_current_generation(tenant_id=_R2_TENANT, run_id=run_id) == 2

    db_path = tmp_path / "retry-resume.db"
    store_a = SQLiteTaskCheckpointStore(db_path=db_path)
    narrow = ParentExecutionAuthority.scoped(("read",))
    saved = store_a.save(
        _paused_checkpoint(
            run_id=run_id,
            attempt_id=attempt_two,
            execution_authority=narrow,
        ),
    )
    assert saved.revision == 1

    identity = resolve_root_task_identity(resume_checkpoint=saved)
    assert identity.run_id == run_id
    assert identity.attempt_id == attempt_two
    assert lifecycle.get_current_generation(tenant_id=_R2_TENANT, run_id=run_id) == 2

    store_b = SQLiteTaskCheckpointStore(db_path=db_path)
    loaded = store_b.get_latest(saved.task_id, _R2_TENANT)
    assert loaded is not None
    restored_task = _resume_task(
        loaded,
        execution_authority=ParentExecutionAuthority.scoped(("read",)),
    )
    restored = LongRunningCoordinator.restore_if_resuming(restored_task, store_b)
    assert restored is not None
    after = lifecycle.get_current_generation(tenant_id=_R2_TENANT, run_id=run_id)
    assert after == 2

    next_ckpt = store_b.save(
        loaded.model_copy(update={"checkpoint_id": "after-resume", "progress_message": "ok"}),
        expected_revision=loaded.revision,
    )
    assert next_ckpt.revision == 2


def test_composite_e2e_2_resume_then_retry_mints_only_via_lifecycle(tmp_path: Path) -> None:
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    retry = ExecutionAttemptRetryService(lifecycle)
    run_id = mint_run_id()
    attempt_one = mint_attempt_id()
    lifecycle.record_initial_attempt(tenant_id=_R2_TENANT, run_id=run_id, attempt_id=attempt_one)
    checkpoint = _paused_checkpoint(run_id=run_id, attempt_id=attempt_one)
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "resume-retry.db")
    saved = store.save(checkpoint)
    identity = resolve_root_task_identity(resume_checkpoint=saved)
    assert identity.attempt_id == attempt_one
    gen_before = lifecycle.get_current_generation(tenant_id=_R2_TENANT, run_id=run_id)
    assert gen_before == 1

    request = ExecutionRetryEligibilityRequest(
        classification=classify_execution_failure(kind=ExecutionFailureKind.RETRYABLE_TRANSIENT),
        attempt_number=gen_before,
        max_attempts=3,
        global_deadline_monotonic=1000.0,
        now_monotonic=10.0,
        proposed_backoff_seconds=1.0,
    )
    transition = retry.transition_for_retry(
        tenant_id=_R2_TENANT,
        task_id=TaskId(saved.task_id),
        run_id=run_id,
        expected_attempt_id=attempt_one,
        request=request,
    )
    assert transition is not None
    assert transition.active_attempt_id != attempt_one
    assert lifecycle.get_current_generation(tenant_id=_R2_TENANT, run_id=run_id) == 2


def test_composite_e2e_6_checkpoint_policy_revoked_blocks_resume() -> None:
    checkpoint = _paused_checkpoint()
    result = evaluate_checkpoint_resume_eligibility(
        checkpoint,
        target_task_id=checkpoint.task_id,
        target_tenant_id=_R2_TENANT,
        policy_decision=PolicyDecision(action=PolicyAction.DENY, reason="revoked"),
    )
    assert result.eligibility is CheckpointResumeEligibility.REJECT_GOVERNANCE


def test_composite_e2e_13_unknown_side_effect_blocks_blind_retry_and_recovery() -> None:
    r1 = evaluate_execution_retry_eligibility(
        ExecutionRetryEligibilityRequest(
            classification=classify_execution_failure(
                kind=ExecutionFailureKind.UNKNOWN_UNSAFE,
                has_unknown_side_effect=True,
            ),
            attempt_number=1,
            max_attempts=3,
        ),
    )
    assert r1.action is ExecutionRetryAction.FAIL
    r3 = evaluate_slot_recovery_policy(
        SlotRecoveryPolicyRequest(
            disposition=SlotRecoveryDisposition.FAILED,
            failure_kind=ExecutionFailureKind.UNKNOWN_UNSAFE,
            has_unknown_side_effect=True,
        ),
    )
    assert r3.action is SlotRecoveryPolicyAction.PRESERVE_FAILURE


def test_composite_e2e_14_hitl_not_automatic_recovery() -> None:
    result = evaluate_slot_recovery_policy(
        SlotRecoveryPolicyRequest(
            disposition=SlotRecoveryDisposition.WAITING_FOR_HUMAN,
            waiting_for_human=True,
        ),
    )
    assert result.action is SlotRecoveryPolicyAction.WAIT


def test_composite_e2e_15_terminal_blocks_resume(tmp_path: Path) -> None:
    checkpoint = _paused_checkpoint()
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "terminal-5e.db")
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
    loaded = store.get_latest(checkpoint.task_id, _R2_TENANT)
    assert loaded is not None
    result = evaluate_checkpoint_resume_eligibility(
        loaded,
        target_task_id=loaded.task_id,
        target_tenant_id=_R2_TENANT,
        execution_terminal=terminal,
    )
    assert result.eligibility is CheckpointResumeEligibility.REJECT_TERMINAL


def test_composite_e2e_12_stale_checkpoint_writer_blocked(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "stale-5e.db")
    base = store.save(_paused_checkpoint())
    store.save(
        base.model_copy(update={"checkpoint_id": "n-plus-1"}),
        expected_revision=base.revision,
    )
    with pytest.raises(StaleCheckpointWriteError):
        store.save(
            base.model_copy(update={"checkpoint_id": "stale-writer"}),
            expected_revision=base.revision,
        )


def test_revision_distinct_from_attempt() -> None:
    checkpoint = _paused_checkpoint()
    assert checkpoint.runtime is not None
    assert checkpoint.revision is None
    sig = inspect.signature(SQLiteTaskCheckpointStore.save)
    assert "expected_revision" in sig.parameters


def test_static_no_reflection_on_recovery_surfaces() -> None:
    surfaces: list[str] = [
        "intergrax/contracts/execution_retry.py",
        *list(_R2_PRODUCTION_SURFACE),
        *list(_R3_PRODUCTION_SURFACE),
    ]
    for relative in surfaces:
        path = _REPO_ROOT / relative
        if path.is_dir():
            for py in path.rglob("*.py"):
                assert _REFLECTION_PATTERN.search(py.read_text(encoding="utf-8")) is None
        else:
            assert _REFLECTION_PATTERN.search(path.read_text(encoding="utf-8")) is None


def test_static_architecture_drift_no_unauthorized_mint_retry() -> None:
    allowed = frozenset(
        {
            "intergrax/runtime/execution/identity_authority.py",
            "intergrax/runtime/execution/attempt_lifecycle/service.py",
        },
    )
    violations: list[str] = []
    for root in (_REPO_ROOT / "intergrax/runtime/execution", _REPO_ROOT / "intergrax/runtime/nexus"):
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(_REPO_ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else None)
                if name == "mint_retry_attempt_id" and rel not in allowed:
                    violations.append(f"{rel}:{node.lineno}")
    assert violations == []


def test_pre_existing_cancellation_fixture_same_root_cause() -> None:
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


def test_pre_existing_partial_results_fixture_unchanged_baseline() -> None:
    drift = subprocess.run(
        ["git", "diff", "--name-only", f"{_R3_FINAL_SHA}..HEAD", "--", "intergrax/"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "intergrax/runtime/long_running/partial_results.py" not in drift.stdout
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


def _static_check_targets() -> list[str]:
    targets: list[str] = []
    for relative in _R1_PRODUCTION_SURFACE:
        path = _REPO_ROOT / relative
        if path.is_dir():
            targets.extend(str(p) for p in path.rglob("*.py"))
        else:
            targets.append(str(path))
    for relative in (
        "intergrax/runtime/long_running/checkpoint_resume_validation.py",
        "intergrax/runtime/long_running/runtime_checkpoint.py",
        "intergrax/runtime/long_running/store.py",
        *list(_R3_PRODUCTION_SURFACE),
    ):
        path = _REPO_ROOT / relative
        if path.is_dir():
            targets.extend(str(p) for p in path.rglob("*.py"))
        else:
            targets.append(str(path))
    targets.append(str(Path(__file__)))
    return targets


def test_ruff_recovery_surfaces_and_final_test() -> None:
    targets = _static_check_targets()
    proc = subprocess.run(
        ["uv", "run", "ruff", "check", *targets],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_pyright_recovery_surfaces_and_final_test() -> None:
    targets = _static_check_targets()
    proc = subprocess.run(
        ["uv", "run", "pyright", *targets],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_composite_e2e_3_4_5_delegated_to_r3_final_gate() -> None:
    proc = _run_pytest(
        [
            "tests/unit/runtime/architecture/test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py::test_final_single_failure_four_slot_recover_c_only",
            "tests/unit/runtime/architecture/test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py::test_final_r1_interop_transient_retry_distinct_from_r3",
            "tests/unit/runtime/architecture/test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py::test_final_unknown_side_effect_blocked",
        ],
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_composite_e2e_7_8_9_10_11_delegated_to_r3_and_r2_final() -> None:
    proc = _run_pytest(
        [
            "tests/unit/runtime/architecture/test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py::test_final_authority_narrowing_scenario",
            "tests/unit/runtime/architecture/test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py::test_final_authority_missing_scenario",
            "tests/unit/runtime/architecture/test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py::test_final_trust_denied_blocked",
            "tests/unit/runtime/architecture/test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py::test_final_parent_cancel_blocked",
            "tests/unit/runtime/architecture/test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py::test_final_same_slot_recovery_idempotent",
        ],
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

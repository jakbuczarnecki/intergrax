# © Artur Czarnecki. All rights reserved.

"""STATE-X-R4 — TaskCheckpoint restore integrity matrix (R4-Q01..Q40)."""

from __future__ import annotations

import ast
from pathlib import Path
import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import mint_task_id
from intergrax.runtime.execution.execution_terminal.persistence import (
    terminal_capability_from_task_checkpoint_store,
)
from intergrax.runtime.long_running.checkpoint_resume_validation import (
    CheckpointResumeEligibility,
    CheckpointResumeValidationError,
    assert_checkpoint_resume_materialization_eligible,
    evaluate_checkpoint_resume_eligibility,
    evaluate_checkpoint_resume_materialization,
    narrow_resume_execution_authority,
    resolve_resume_execution_authority,
    validate_checkpoint_resume_authority,
)
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.resume_planner import (
    _base_resume_task,
    build_checkpoint_resume_task,
)
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.task.task import Task, TaskState
from tests.qualification.state_x._r4_task_checkpoint_restore_support import (
    RESTORE_CONSUMER_INVENTORY,
    STATE_X_R4_PRE_AUDIT_HEAD,
    resume_planner_validates_before_model_validate,
    scheduler_validates_before_build,
)
from tests.qualification.state_x.test_state_x_r1_checkpoint_resume_terminal import (
    _paused_checkpoint,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_r4_q01_closed_world_restore_consumer_inventory() -> None:
    assert len(RESTORE_CONSUMER_INVENTORY) >= 8
    for _name, rel_path, _note in RESTORE_CONSUMER_INVENTORY:
        assert (_REPO_ROOT / rel_path).is_file()


def test_r4_q02_task_checkpoint_persistence_single_owner() -> None:
    src = (
        _REPO_ROOT / "intergrax/runtime/long_running/persistence_contract.py"
    ).read_text(encoding="utf-8")
    assert "class TaskCheckpointPersistence" in src
    assert "RestoreCheckpointStore" not in src


def test_r4_q03_single_canonical_restore_validator_owner() -> None:
    path = _REPO_ROOT / "intergrax/runtime/long_running/checkpoint_resume_validation.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert any(
        isinstance(n, ast.FunctionDef) and n.name == "assert_checkpoint_resume_eligible"
        for n in tree.body
    )
    planner = (
        _REPO_ROOT / "intergrax/runtime/long_running/resume_planner.py"
    ).read_text(encoding="utf-8")
    assert "def validate_checkpoint" not in planner


def test_r4_q04_no_task_materialization_before_restore_validation() -> None:
    assert resume_planner_validates_before_model_validate()
    assert scheduler_validates_before_build()


def test_r4_q05_empty_task_snapshot_rejected() -> None:
    cp = _paused_checkpoint()
    cp = cp.model_copy(update={"task_snapshot": {}})
    with pytest.raises(CheckpointResumeValidationError):
        assert_checkpoint_resume_materialization_eligible(
            cp,
            target_task_id=cp.task_id,
            target_tenant_id=cp.tenant_id,
        )


def test_r4_q06_missing_task_id_in_snapshot_rejected() -> None:
    cp = _paused_checkpoint()
    snap = dict(cp.task_snapshot)
    snap.pop("task_id", None)
    cp = cp.model_copy(update={"task_snapshot": snap})
    with pytest.raises(CheckpointResumeValidationError):
        assert_checkpoint_resume_materialization_eligible(
            cp,
            target_task_id=cp.task_id,
            target_tenant_id=cp.tenant_id,
        )


def test_r4_q07_missing_tenant_id_in_snapshot_rejected() -> None:
    cp = _paused_checkpoint()
    snap = dict(cp.task_snapshot)
    snap.pop("tenant_id", None)
    cp = cp.model_copy(update={"task_snapshot": snap})
    with pytest.raises(CheckpointResumeValidationError):
        assert_checkpoint_resume_materialization_eligible(
            cp,
            target_task_id=cp.task_id,
            target_tenant_id=cp.tenant_id,
        )


def test_r4_q08_snapshot_task_mismatch_rejected() -> None:
    cp = _paused_checkpoint()
    other = str(mint_task_id())
    snap = dict(cp.task_snapshot)
    snap["task_id"] = other
    cp = cp.model_copy(update={"task_snapshot": snap})
    with pytest.raises(CheckpointResumeValidationError):
        assert_checkpoint_resume_materialization_eligible(
            cp,
            target_task_id=cp.task_id,
            target_tenant_id=cp.tenant_id,
        )


def test_r4_q09_snapshot_tenant_mismatch_rejected() -> None:
    cp = _paused_checkpoint()
    snap = dict(cp.task_snapshot)
    snap["tenant_id"] = "tenant-forged"
    cp = cp.model_copy(update={"task_snapshot": snap})
    with pytest.raises(CheckpointResumeValidationError):
        assert_checkpoint_resume_materialization_eligible(
            cp,
            target_task_id=cp.task_id,
            target_tenant_id=cp.tenant_id,
        )


def test_r4_q10_unsupported_task_checkpoint_schema_rejected() -> None:
    cp = _paused_checkpoint()
    cp = cp.model_copy(update={"schema_version": "task_checkpoint.v99"})
    result = evaluate_checkpoint_resume_materialization(
        cp,
        target_task_id=cp.task_id,
        target_tenant_id=cp.tenant_id,
    )
    assert result.eligibility is CheckpointResumeEligibility.REJECT_SCHEMA


def test_r4_q11_unsupported_runtime_checkpoint_schema_rejected() -> None:
    cp = _paused_checkpoint()
    assert cp.runtime is not None
    runtime = cp.runtime.model_copy(update={"schema_version": "runtime.v99"})
    cp = cp.model_copy(update={"runtime": runtime})
    result = evaluate_checkpoint_resume_materialization(
        cp,
        target_task_id=cp.task_id,
        target_tenant_id=cp.tenant_id,
    )
    assert result.eligibility is CheckpointResumeEligibility.REJECT_SCHEMA


def test_r4_q13_stale_checkpoint_rejected() -> None:
    older = _paused_checkpoint(revision=1)
    newer = older.model_copy(
        update={
            "revision": 2,
            "resume_token": "rt-new",
            "checkpoint_id": f"{older.checkpoint_id}-superseded",
        }
    )
    with pytest.raises(CheckpointResumeValidationError):
        assert_checkpoint_resume_materialization_eligible(
            older,
            target_task_id=older.task_id,
            target_tenant_id=older.tenant_id,
            latest_checkpoint=newer,
        )


def test_r4_q14_tenant_a_checkpoint_tenant_b_resume_denied() -> None:
    cp = _paused_checkpoint(tenant_id="tenant-a")
    with pytest.raises(CheckpointResumeValidationError):
        assert_checkpoint_resume_materialization_eligible(
            cp,
            target_task_id=cp.task_id,
            target_tenant_id="tenant-b",
        )


def test_r4_q15_task_a_checkpoint_task_b_resume_denied() -> None:
    cp = _paused_checkpoint()
    with pytest.raises(CheckpointResumeValidationError):
        assert_checkpoint_resume_materialization_eligible(
            cp,
            target_task_id=str(mint_task_id()),
            target_tenant_id=cp.tenant_id,
        )


def test_r4_q16_historical_authority_cannot_mint_without_current() -> None:
    broad = ParentExecutionAuthority.unrestricted_root()
    cp = _paused_checkpoint(execution_authority=broad)
    result = validate_checkpoint_resume_authority(cp, None)
    assert result.eligibility is CheckpointResumeEligibility.REJECT_AUTHORITY


def test_r4_q17_resume_authority_cannot_exceed_current() -> None:
    cp = _paused_checkpoint(
        execution_authority=ParentExecutionAuthority.scoped(("admin",)),
    )
    result = validate_checkpoint_resume_authority(
        cp,
        Task(
            task_id=cp.task_id,
            tenant_id=cp.tenant_id,
            user_id="u",
            message="m",
            state=TaskState.WAITING_FOR_HUMAN,
            execution_authority=None,
        ),
    )
    assert result.eligibility is CheckpointResumeEligibility.REJECT_AUTHORITY


def test_r4_q18_narrowed_resume_within_historical_bound() -> None:
    historical = ParentExecutionAuthority.scoped(("read", "write"))
    current = ParentExecutionAuthority.scoped(("read", "write", "admin"))
    narrowed = narrow_resume_execution_authority(current, historical)
    assert narrowed.permission_scopes == ("read", "write")


def test_r4_q19_terminal_checkpoint_state_rejected() -> None:
    cp = _paused_checkpoint()
    cp = cp.model_copy(update={"task_state": TaskState.COMPLETED})
    result = evaluate_checkpoint_resume_materialization(
        cp,
        target_task_id=cp.task_id,
        target_tenant_id=cp.tenant_id,
    )
    assert result.eligibility is CheckpointResumeEligibility.REJECT_STATE


def test_r4_q24_scheduler_validates_before_task_materialization() -> None:
    assert scheduler_validates_before_build()


def test_r4_q27_operator_resume_validates_before_materialization() -> None:
    text = (
        _REPO_ROOT / "intergrax/applications/_shared/task_control.py"
    ).read_text(encoding="utf-8")
    assert "assert_checkpoint_resume_materialization_eligible" in text
    idx = text.find("def governed_resume_checkpoint_task")
    segment = text[idx : idx + 4000]
    assert segment.find("assert_checkpoint_resume_materialization_eligible") < segment.find(
        "_validate_operator_hitl_input"
    )


def test_r4_q32_worker_preserves_incoming_tenant() -> None:
    src = (
        _REPO_ROOT / "intergrax/runtime/task/nexus_worker_execution.py"
    ).read_text(encoding="utf-8")
    assert "tenant_id=execution_identity.tenant_id" in src


def test_r4_q35_q2_d1_terminal_capability_helper() -> None:
    store = SQLiteTaskCheckpointStore(db_path=":memory:")
    cap = terminal_capability_from_task_checkpoint_store(store)
    assert cap is store
    assert terminal_capability_from_task_checkpoint_store(object()) is None


def test_r4_q36_restore_semantics_not_sqlite_specific() -> None:
    cp = _paused_checkpoint()
    with pytest.raises(CheckpointResumeValidationError):
        build_checkpoint_resume_task(
            cp.model_copy(update={"task_snapshot": {}}),
            target_task_id=cp.task_id,
            target_tenant_id=cp.tenant_id,
        )


def test_r4_q37_r3_r4_scheduler_regression_import() -> None:
    from tests.qualification.state_x import _r3_r4_qualification_tests as r34  # noqa: F401

    assert r34 is not None


def test_r4_q38_r3_r5_checkpoint_regression_import() -> None:
    from tests.qualification.state_x import _r3_r5_qualification_tests as r35  # noqa: F401

    assert r35 is not None


def test_r4_q40_tenant_isolation_adversarial_no_cross_tenant_materialization() -> None:
    cp = _paused_checkpoint(tenant_id="tenant-a")
    snap = dict(cp.task_snapshot)
    snap["tenant_id"] = "tenant-b"
    cp = cp.model_copy(update={"task_snapshot": snap})
    with pytest.raises(CheckpointResumeValidationError):
        _base_resume_task(
            cp,
            target_task_id=cp.task_id,
            target_tenant_id=cp.tenant_id,
        )


def test_r4_pre_audit_head_constant() -> None:
    assert STATE_X_R4_PRE_AUDIT_HEAD == "716ed746f8b463681db536804235cedc86adc162"

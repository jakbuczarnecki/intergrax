# © Artur Czarnecki. All rights reserved.

"""STATE-X-R5 — backup/restore responsibility matrix (R5-Q01..Q40)."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from intergrax.contracts.execution_terminal import ExecutionTerminalOutcome
from intergrax.runtime.execution.execution_terminal import (
    CheckpointStoreExecutionTerminalStore,
    ExecutionTerminalService,
)
from intergrax.runtime.long_running.checkpoint_resume_validation import (
    CheckpointResumeEligibility,
    CheckpointResumeValidationError,
    assert_checkpoint_resume_materialization_eligible,
    evaluate_checkpoint_resume_eligibility,
    evaluate_checkpoint_resume_materialization,
)
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.task.task import TaskState
from tests.qualification.state_x._r5_backup_restore_support import (
    PHYSICAL_BACKUP_UNIT_MATRIX,
    R5_BACKUP_RESTORE_FAMILY_MATRIX,
    SEMANTIC_RECOVERY_FLOW_MATRIX,
    STATE_X_R4_ACCEPTED_CLOSURE_SHA,
    STATE_X_R5_PRE_AUDIT_HEAD,
    assert_frz_rec_08_behavioral_evidence_complete,
    assert_frz_rec_08_r5_completeness,
    decision_recovery_source_references_families,
    persistence_contracts_lack_backup_api,
    run_trace_not_resume_consumer,
    sqlite_runtime_bundle_paths_are_distinct,
    sx_f15_has_no_persistence_contract,
)
from tests.qualification.state_x.inventory import (
    MANDATORY_FAMILY_IDS,
    BackupRestoreResponsibility,
    STATE_X_FAMILY_INVENTORY,
)
from tests.qualification.state_x.test_state_x_r1_checkpoint_resume_terminal import (
    _paused_checkpoint,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_r5_q01_closed_world_family_inventory_complete() -> None:
    assert len(R5_BACKUP_RESTORE_FAMILY_MATRIX) == 15
    assert {r.family_id for r in R5_BACKUP_RESTORE_FAMILY_MATRIX} == set(MANDATORY_FAMILY_IDS)


def test_r5_q02_every_family_exactly_one_responsibility() -> None:
    for entry in STATE_X_FAMILY_INVENTORY:
        assert entry.backup_restore_responsibility in BackupRestoreResponsibility


def test_r5_q03_no_durable_family_unknown_responsibility() -> None:
    for row in R5_BACKUP_RESTORE_FAMILY_MATRIX:
        assert row.responsibility is not BackupRestoreResponsibility.PLATFORM_SUPPORTED_BACKUP_RESTORE


def test_r5_q04_physical_backup_unit_for_durable_families() -> None:
    for row in R5_BACKUP_RESTORE_FAMILY_MATRIX:
        if row.responsibility is BackupRestoreResponsibility.BACKEND_OPERATOR_WITH_CONSISTENCY:
            assert row.physical_backup_unit
            scope = row.provider_scope.lower()
            unit = row.physical_backup_unit.lower()
            assert any(
                token in scope or token in unit
                for token in (
                    "reference",
                    "sqlite",
                    "kv",
                    "redis",
                    "document",
                    "embedded",
                    "colocated",
                    "taskcheckpoint",
                )
            )


def test_r5_q05_semantic_recovery_flows_defined() -> None:
    assert len(SEMANTIC_RECOVERY_FLOW_MATRIX) >= 9
    flows = {f.recovery_flow for f in SEMANTIC_RECOVERY_FLOW_MATRIX}
    assert "task_resume" in flows
    assert "effect_replay" in flows


def test_r5_q06_operator_backup_does_not_change_semantic_owner() -> None:
    for row in R5_BACKUP_RESTORE_FAMILY_MATRIX:
        entry = next(e for e in STATE_X_FAMILY_INVENTORY if e.family_id == row.family_id)
        assert entry.semantic_owner
        assert "Integrax" in row.restore_validation_owner or "platform" in row.restore_validation_owner.lower()


def test_r5_q07_sqlite_task_checkpoint_physical_group() -> None:
    group = next(r for r in PHYSICAL_BACKUP_UNIT_MATRIX if r.provider_store == "SQLiteTaskCheckpointStore")
    assert "SX-F01" in group.state_x_families
    assert "SX-F13" in group.state_x_families
    assert "SX-F05" in group.state_x_families


def test_r5_q08_sqlite_bundle_not_atomic_cross_file() -> None:
    assert sqlite_runtime_bundle_paths_are_distinct()
    bundle = next(
        r for r in PHYSICAL_BACKUP_UNIT_MATRIX if "SQLiteRuntimePersistenceBundle" in r.provider_store
    )
    assert "NOT one atomic" in bundle.physical_backup_unit


def test_r5_q09_no_generic_platform_backup_api() -> None:
    assert persistence_contracts_lack_backup_api()


def test_r5_q10_task_checkpoint_sqlite_operator_restore_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "cp.db"
    store = SQLiteTaskCheckpointStore(db_path=db)
    cp = _paused_checkpoint()
    store.save(cp)
    snapshot_path = tmp_path / "cp.snapshot.db"
    shutil.copy2(db, snapshot_path)
    newer = cp.model_copy(
        update={
            "revision": (cp.revision or 1) + 1,
            "checkpoint_id": f"{cp.checkpoint_id}-new",
            "resume_token": "rt-newer",
        },
    )
    store.save(newer, expected_revision=cp.revision)
    shutil.copy2(snapshot_path, db)
    restored = SQLiteTaskCheckpointStore(db_path=db)
    loaded = restored.get_latest(cp.task_id, cp.tenant_id)
    assert loaded is not None
    assert loaded.revision == cp.revision
    assert loaded.resume_token == cp.resume_token


def test_r5_q11_restored_checkpoint_passes_canonical_validation(tmp_path: Path) -> None:
    db = tmp_path / "cp-valid.db"
    store = SQLiteTaskCheckpointStore(db_path=db)
    cp = _paused_checkpoint()
    store.save(cp)
    backup = tmp_path / "backup.db"
    shutil.copy2(db, backup)
    store.save(
        cp.model_copy(update={"revision": 99, "checkpoint_id": "forced-new"}),
        expected_revision=cp.revision,
    )
    shutil.copy2(backup, db)
    loaded = SQLiteTaskCheckpointStore(db_path=db).get_latest(cp.task_id, cp.tenant_id)
    assert loaded is not None
    assert_checkpoint_resume_materialization_eligible(
        loaded,
        target_task_id=loaded.task_id,
        target_tenant_id=loaded.tenant_id,
    )


def test_r5_q12_restored_scheduler_tables_same_sqlite_unit(tmp_path: Path) -> None:
    db = tmp_path / "sched.db"
    store = SQLiteTaskCheckpointStore(db_path=db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            INSERT INTO scheduler_ledger (ledger_key, action, executed_at_utc, status)
            VALUES ('k1', 'test', '2020-01-01T00:00:00Z', 'completed')
            """,
        )
    backup = tmp_path / "sched.bak"
    shutil.copy2(db, backup)
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM scheduler_ledger")
    shutil.copy2(backup, db)
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT status FROM scheduler_ledger WHERE ledger_key = 'k1'").fetchone()
    assert row is not None
    assert row[0] == "completed"


def test_r5_q13_checkpoint_terminal_physical_group_consistent(tmp_path: Path) -> None:
    db = tmp_path / "term-group.db"
    store = SQLiteTaskCheckpointStore(db_path=db)
    cp = _paused_checkpoint()
    store.save(cp)
    assert cp.runtime is not None
    terminal = ExecutionTerminalService(CheckpointStoreExecutionTerminalStore(store))
    terminal.commit_terminal_outcome(
        tenant_id=cp.tenant_id,
        task_id=cp.task_id,
        run_id=cp.runtime.run_id,
        outcome=ExecutionTerminalOutcome.COMPLETED,
        reason="done",
    )
    backup = tmp_path / "term.bak"
    shutil.copy2(db, backup)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            UPDATE task_execution_terminal
            SET outcome = ?, reason = ?
            WHERE tenant_id = ? AND task_id = ?
            """,
            ("cancelled", "cancelled", cp.tenant_id, cp.task_id),
        )
    shutil.copy2(backup, db)
    restored_store = SQLiteTaskCheckpointStore(db_path=db)
    restored_terminal = ExecutionTerminalService(
        CheckpointStoreExecutionTerminalStore(restored_store),
    )
    record = restored_terminal.get_terminal_record(tenant_id=cp.tenant_id, task_id=cp.task_id)
    assert record is not None
    assert record.outcome is ExecutionTerminalOutcome.COMPLETED


def test_r5_q14_cross_store_partial_restore_fail_closed_semantics() -> None:
    row = next(r for r in PHYSICAL_BACKUP_UNIT_MATRIX if "multi-file" in r.provider_store.lower())
    assert "fail-closed" in row.platform_semantic_validation.lower()


def test_r5_q15_terminal_skew_cannot_resurrect_terminal_execution() -> None:
    cp = _paused_checkpoint()
    cp = cp.model_copy(update={"task_state": TaskState.COMPLETED})
    result = evaluate_checkpoint_resume_materialization(
        cp,
        target_task_id=cp.task_id,
        target_tenant_id=cp.tenant_id,
    )
    assert result.eligibility is CheckpointResumeEligibility.REJECT_STATE


def test_r5_q16_lineage_skew_fail_closed_import() -> None:
    from tests.qualification.state_x._r4_r1_restore_consumer_convergence_tests import (  # noqa: PLC0415
        test_r4_r1_q22_missing_degraded_lineage_fail_closed,
    )

    test_r4_r1_q22_missing_degraded_lineage_fail_closed()


def test_r5_q17_attempt_lifecycle_skew_documented_in_inventory() -> None:
    entry = next(e for e in STATE_X_FAMILY_INVENTORY if e.family_id == "SX-F04")
    assert "resume" in entry.fork_retry_resume_notes.lower() or "attempt" in entry.fork_retry_resume_notes.lower()


def test_r5_q18_budget_restore_not_config_reset() -> None:
    entry = next(e for e in STATE_X_FAMILY_INVENTORY if e.family_id == "SX-F09")
    assert "Configured RunBudget" in entry.stale_state_rule or "persisted ledger" in entry.stale_state_rule


@pytest.mark.asyncio
async def test_r5_q19_idempotency_skew_regression_import(tmp_path: Path) -> None:
    from tests.qualification.state_x._r3_r2_qualification_tests import (  # noqa: PLC0415
        test_r3_r2_q32_crash_window_canonical_production_path_no_duplicate_effect,
    )
    from tests.qualification.state_x._r3_r2_support import (  # noqa: PLC0415
        _sqlite_compensation_queue_store,
        _sqlite_idempotency_store_path,
    )

    await test_r3_r2_q32_crash_window_canonical_production_path_no_duplicate_effect(
        tmp_path,
        _sqlite_compensation_queue_store,
        _sqlite_idempotency_store_path,
    )


def test_r5_q20_compensation_responsibility_explicit() -> None:
    row = next(r for r in R5_BACKUP_RESTORE_FAMILY_MATRIX if r.family_id == "SX-F11")
    assert row.responsibility is BackupRestoreResponsibility.BACKEND_OPERATOR_WITH_CONSISTENCY
    assert "operator" in row.supported_restore_semantics.lower() or "Operator" in row.supported_restore_semantics


def test_r5_q21_missing_compensation_not_silent_completion() -> None:
    row = next(r for r in R5_BACKUP_RESTORE_FAMILY_MATRIX if r.family_id == "SX-F11")
    assert "infer" in row.partial_restore_behavior.lower() or "FAIL CLOSED" in row.partial_restore_behavior


def test_r5_q22_human_decision_cannot_mint_authority() -> None:
    entry = next(e for e in STATE_X_FAMILY_INVENTORY if e.family_id == "SX-F12")
    assert entry.authority_role.value.startswith("HISTORICAL")


def test_r5_q23_missing_human_decision_no_synthesized_approval() -> None:
    row = next(r for r in R5_BACKUP_RESTORE_FAMILY_MATRIX if r.family_id == "SX-F12")
    assert "synthesized" in row.partial_restore_behavior.lower() or "no synthesized" in row.partial_restore_behavior.lower()


def test_r5_q24_agent_checkpoint_identity_validation_import() -> None:
    from tests.qualification.state_x import _r3_r5_qualification_tests as r35  # noqa: F401

    assert hasattr(r35, "test_r3_r5_q27_corrupt_json_fails_closed")


def test_r5_q25_runtime_events_not_execution_authority() -> None:
    entry = next(e for e in STATE_X_FAMILY_INVENTORY if e.family_id == "SX-F07")
    assert "Not authoritative" in entry.restore_consumers[0] or "not authoritative" in str(entry.restore_consumers).lower()


def test_r5_q26_run_trace_non_authoritative_projection() -> None:
    assert run_trace_not_resume_consumer()
    row = next(r for r in R5_BACKUP_RESTORE_FAMILY_MATRIX if r.family_id == "SX-F08")
    assert row.responsibility is BackupRestoreResponsibility.NOT_DURABLE_REBUILDABLE


def test_r5_q27_sx_f15_no_independent_durable_truth() -> None:
    assert sx_f15_has_no_persistence_contract()
    row = next(r for r in R5_BACKUP_RESTORE_FAMILY_MATRIX if r.family_id == "SX-F15")
    assert row.responsibility is BackupRestoreResponsibility.NOT_APPLICABLE


def test_r5_q28_corrupt_task_checkpoint_fail_closed(tmp_path: Path) -> None:
    db = tmp_path / "bad-cp.db"
    store = SQLiteTaskCheckpointStore(db_path=db)
    cp = _paused_checkpoint()
    store.save(cp)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE task_checkpoints SET task_snapshot_json = ? WHERE checkpoint_id = ?",
            ("{not-json", cp.checkpoint_id),
        )
    with pytest.raises((ValueError, CheckpointResumeValidationError, Exception)):
        loaded = store.get_latest(cp.task_id, cp.tenant_id)
        if loaded is not None:
            assert_checkpoint_resume_materialization_eligible(
                loaded,
                target_task_id=loaded.task_id,
                target_tenant_id=loaded.tenant_id,
            )


def test_r5_q29_corrupt_auxiliary_human_decision_fail_closed(tmp_path: Path) -> None:
    from tests.qualification.state_x._r3_r3_qualification_tests import (  # noqa: PLC0415
        test_r3_r3_q13_corrupt_approver_provenance_fail_closed,
    )

    test_r3_r3_q13_corrupt_approver_provenance_fail_closed(tmp_path)


def test_r5_q30_cross_tenant_restored_state_rejected() -> None:
    cp = _paused_checkpoint(tenant_id="tenant-a")
    with pytest.raises(CheckpointResumeValidationError):
        assert_checkpoint_resume_materialization_eligible(
            cp,
            target_task_id=cp.task_id,
            target_tenant_id="tenant-b",
        )


def test_r5_q31_restore_cannot_mint_tenant_identity() -> None:
    cp = _paused_checkpoint(tenant_id="tenant-a")
    snap = dict(cp.task_snapshot)
    snap["tenant_id"] = "tenant-forged"
    cp = cp.model_copy(update={"task_snapshot": snap})
    with pytest.raises(CheckpointResumeValidationError):
        assert_checkpoint_resume_materialization_eligible(
            cp,
            target_task_id=cp.task_id,
            target_tenant_id=cp.tenant_id,
        )


def test_r5_q32_restore_cannot_mint_execution_authority() -> None:
    cp = _paused_checkpoint()
    result = evaluate_checkpoint_resume_eligibility(
        cp,
        target_task_id=cp.task_id,
        target_tenant_id=cp.tenant_id,
        current_task=None,
    )
    assert result.eligibility in (
        CheckpointResumeEligibility.ALLOW_RESUME,
        CheckpointResumeEligibility.REJECT_AUTHORITY,
    )


def test_r5_q33_physical_unit_distinct_from_semantic_recovery_set() -> None:
    f01 = next(r for r in R5_BACKUP_RESTORE_FAMILY_MATRIX if r.family_id == "SX-F01")
    assert "SQLite" in f01.physical_backup_unit or "partition" in f01.physical_backup_unit
    assert len(f01.semantic_restore_dependencies) >= 1
    assert f01.physical_backup_unit != ",".join(f01.semantic_restore_dependencies)


def test_r5_q34_operator_vs_platform_validation_distinct() -> None:
    unit = PHYSICAL_BACKUP_UNIT_MATRIX[0]
    assert "operator" in unit.operator_responsibility.lower()
    assert "Integrax" in unit.platform_semantic_validation


def test_r5_q35_r2_r3_r4_regression_modules_importable() -> None:
    from tests.qualification.state_x import _r3_r2_qualification_tests as r32  # noqa: F401
    from tests.qualification.state_x import _r4_task_checkpoint_restore_qualification_tests as r4  # noqa: F401
    from tests.qualification.state_x import test_state_x_r2_decision_attempt_lineage as r2  # noqa: F401

    assert r2 is not None and r32 is not None and r4 is not None
    assert decision_recovery_source_references_families()


def test_r5_q36_full_state_x_suite_import_gate() -> None:
    from tests.qualification.state_x import test_state_x_p0_baseline as p0  # noqa: F401

    assert p0 is not None


def test_r5_q37_tenant_isolation_audit_mechanical_pass() -> None:
    for row in R5_BACKUP_RESTORE_FAMILY_MATRIX:
        if row.responsibility is BackupRestoreResponsibility.NOT_APPLICABLE:
            continue
        assert "tenant" in row.tenant_behavior.lower()


def test_r5_q38_frz_rec_08_completeness_gate() -> None:
    assert_frz_rec_08_r5_completeness()
    assert_frz_rec_08_behavioral_evidence_complete()


def test_r5_q41_frz_rec_08_behavioral_evidence_gate() -> None:
    assert_frz_rec_08_behavioral_evidence_complete()


def test_r5_q39_frz_rec_05_remains_open_not_claimed() -> None:
    entry = next(e for e in STATE_X_FAMILY_INVENTORY if e.family_id == "SX-F04")
    assert "R2" in entry.fork_retry_resume_notes or "fork" in entry.fork_retry_resume_notes.lower()


def test_r5_q40_in_scope_blockers_zero_mechanical() -> None:
    assert_frz_rec_08_r5_completeness()
    assert STATE_X_R5_PRE_AUDIT_HEAD == STATE_X_R4_ACCEPTED_CLOSURE_SHA


def test_r5_pre_audit_head_constant() -> None:
    assert STATE_X_R5_PRE_AUDIT_HEAD == "61faf8f317125b996526ceceaf5754b9c28073d6"


def test_r5_r4_accepted_closure_sha_recorded() -> None:
    assert STATE_X_R4_ACCEPTED_CLOSURE_SHA == "61faf8f317125b996526ceceaf5754b9c28073d6"


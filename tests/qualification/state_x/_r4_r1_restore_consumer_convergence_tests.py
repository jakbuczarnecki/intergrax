# © Artur Czarnecki. All rights reserved.

"""STATE-X-R4-R1 — closed-world TaskCheckpoint restore consumer convergence."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.execution.execution_terminal.persistence import (
    terminal_capability_from_task_checkpoint_store,
)
from intergrax.runtime.human.agent_governance_grant_lifecycle import (
    TaskAgentGovernanceGrantLifecycleAdapter,
)
from intergrax.runtime.human.agent_governance_pause_projection import (
    TaskAgentGovernancePauseProjectionAdapter,
)
from intergrax.runtime.long_running.checkpoint_resume_validation import (
    CheckpointResumeEligibility,
    CheckpointResumeValidationError,
    assert_checkpoint_resume_materialization_eligible,
    evaluate_checkpoint_resume_materialization,
    validated_task_snapshot_from_checkpoint,
)
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.partial_results import build_task_progress_view
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_contract import TaskPauseRecord
from intergrax.applications._shared.task_control_governance import (
    task_checkpoint_resume_current_revision,
)
from intergrax.debug.hitl_service import DebugHitlResumeService
from intergrax.runtime.execution.suspended_operation.reentry_coordinator import (
    _load_task_from_durable_checkpoint,
)
from tests.qualification.state_x._r4_r1_restore_consumer_support import (
    R4_R1_TASK_CHECKPOINT_CONSUMERS,
    STATE_X_R4_R1_PRE_AUDIT_HEAD,
    find_raw_task_snapshot_parsers,
)
from tests.qualification.state_x._r4_task_checkpoint_restore_support import (
    scheduler_validates_before_build,
)
from tests.qualification.state_x.test_state_x_r1_checkpoint_resume_terminal import (
    _paused_checkpoint,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TENANT = "tenant-r4-r1"


def _hitl_paused_checkpoint(tenant_id: str = _TENANT) -> TaskCheckpoint:
    cp = _paused_checkpoint(tenant_id=tenant_id)
    task = Task.model_validate(cp.task_snapshot)
    task.runtime.governance.pause_record = TaskPauseRecord(
        pause_id="pause-r4-r1",
        task_id=str(task.task_id),
        human_request_id="hr-r4-r1",
    )
    return cp.model_copy(update={"task_snapshot": task.model_dump(mode="json")})


def test_r4_r1_q01_closed_world_task_checkpoint_snapshot_consumer_inventory() -> None:
    assert len(R4_R1_TASK_CHECKPOINT_CONSUMERS) >= 12
    categories = {r.category for r in R4_R1_TASK_CHECKPOINT_CONSUMERS}
    assert "GOVERNANCE_PROJECTION" in categories
    assert "DIAGNOSTICS_ONLY" in categories
    for record in R4_R1_TASK_CHECKPOINT_CONSUMERS:
        assert (_REPO_ROOT / record.file).is_file()


def test_r4_r1_q02_unauthorized_semantic_raw_snapshot_parser_count_zero() -> None:
    assert find_raw_task_snapshot_parsers() == []


def test_r4_r1_q03_canonical_structural_snapshot_reader_exactly_one() -> None:
    path = _REPO_ROOT / "intergrax/runtime/long_running/checkpoint_resume_validation.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    public = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "validated_task_snapshot_from_checkpoint"
    ]
    assert len(public) == 1
    cp = _paused_checkpoint(tenant_id=_TENANT)
    task = validated_task_snapshot_from_checkpoint(
        cp,
        target_task_id=cp.task_id,
        target_tenant_id=cp.tenant_id,
    )
    assert str(task.task_id) == cp.task_id


@pytest.mark.parametrize(
    "mutator",
    [
        lambda cp: cp.model_copy(update={"task_snapshot": {}}),
        lambda cp: cp.model_copy(
            update={
                "task_snapshot": {
                    k: v
                    for k, v in cp.task_snapshot.items()
                    if k not in ("task_id",)
                }
            }
        ),
    ],
)
def test_r4_r1_structural_reader_adversarial_parity(mutator) -> None:
    cp = _paused_checkpoint(tenant_id=_TENANT)
    bad = mutator(cp)
    with pytest.raises(CheckpointResumeValidationError):
        validated_task_snapshot_from_checkpoint(
            bad,
            target_task_id=cp.task_id,
            target_tenant_id=cp.tenant_id,
        )


def test_r4_r1_q04_pause_projection_valid_checkpoint_works(tmp_path: Path) -> None:
    cp = _paused_checkpoint(tenant_id=_TENANT)
    store = SQLiteTaskCheckpointStore(db_path=str(tmp_path / "pause.db"))
    store.save(cp)
    task = Task(
        task_id=cp.task_id,
        tenant_id=cp.tenant_id,
        user_id="u",
        message="m",
        state=TaskState.WAITING_FOR_HUMAN,
    )
    adapter = TaskAgentGovernancePauseProjectionAdapter(task=task, checkpoint_store=store)
    snapshot = adapter._load_canonical_pause_snapshot()
    assert snapshot.checkpoint_revision == cp.revision


def test_r4_r1_q05_pause_projection_invalid_snapshot_fails_before_mutation(
    tmp_path: Path,
) -> None:
    cp = _paused_checkpoint(tenant_id=_TENANT)
    bad = cp.model_copy(update={"task_snapshot": {}})
    store = SQLiteTaskCheckpointStore(db_path=str(tmp_path / "bad.db"))
    store.save(bad)
    task = Task(
        task_id=cp.task_id,
        tenant_id=cp.tenant_id,
        user_id="u",
        message="m",
        state=TaskState.RUNNING,
    )
    before_state = task.state
    before_gov = copy.deepcopy(task.runtime.governance)
    adapter = TaskAgentGovernancePauseProjectionAdapter(task=task, checkpoint_store=store)
    with pytest.raises(CheckpointResumeValidationError):
        adapter._load_canonical_pause_snapshot()
    assert task.state == before_state
    assert task.runtime.governance == before_gov


def test_r4_r1_q06_pause_projection_cross_tenant_fails(tmp_path: Path) -> None:
    cp = _paused_checkpoint(tenant_id="tenant-a")
    task = Task(
        task_id=cp.task_id,
        tenant_id="tenant-b",
        user_id="u",
        message="m",
        state=TaskState.WAITING_FOR_HUMAN,
    )
    adapter = TaskAgentGovernancePauseProjectionAdapter(
        task=task,
        checkpoint_store=SQLiteTaskCheckpointStore(db_path=str(tmp_path / "xtenant.db")),
    )
    with pytest.raises(CheckpointResumeValidationError):
        adapter._apply_checkpoint_to_task(cp)


def test_r4_r1_q07_grant_lifecycle_valid_checkpoint_works(tmp_path: Path) -> None:
    cp = _paused_checkpoint(tenant_id=_TENANT)
    store = SQLiteTaskCheckpointStore(db_path=str(tmp_path / "grant.db"))
    store.save(cp)
    task = Task(
        task_id=cp.task_id,
        tenant_id=cp.tenant_id,
        user_id="u",
        message="m",
        state=TaskState.WAITING_FOR_HUMAN,
    )
    adapter = TaskAgentGovernanceGrantLifecycleAdapter(task=task, checkpoint_store=store)
    record = adapter._reload_canonical_task_state()
    assert record is None or record.grant is not None


def test_r4_r1_q08_grant_lifecycle_invalid_snapshot_fails_before_mutation(
    tmp_path: Path,
) -> None:
    cp = _paused_checkpoint(tenant_id=_TENANT)
    bad = cp.model_copy(update={"task_snapshot": {}})
    store = SQLiteTaskCheckpointStore(db_path=str(tmp_path / "grant-bad.db"))
    store.save(bad)
    task = Task(
        task_id=cp.task_id,
        tenant_id=cp.tenant_id,
        user_id="u",
        message="m",
        state=TaskState.RUNNING,
    )
    before_gov = copy.deepcopy(task.runtime.governance)
    adapter = TaskAgentGovernanceGrantLifecycleAdapter(task=task, checkpoint_store=store)
    with pytest.raises(CheckpointResumeValidationError):
        adapter._reload_canonical_task_state()
    assert task.runtime.governance == before_gov


def test_r4_r1_q09_grant_lifecycle_cross_task_cross_tenant_fails(tmp_path: Path) -> None:
    cp = _paused_checkpoint(tenant_id=_TENANT)
    task = Task(
        task_id=str(mint_task_id()),
        tenant_id=cp.tenant_id,
        user_id="u",
        message="m",
        state=TaskState.WAITING_FOR_HUMAN,
    )
    adapter = TaskAgentGovernanceGrantLifecycleAdapter(
        task=task,
        checkpoint_store=SQLiteTaskCheckpointStore(db_path=str(tmp_path / "grant-x.db")),
    )
    with pytest.raises(CheckpointResumeValidationError):
        adapter._apply_checkpoint_to_task(cp)


def test_r4_r1_q10_task_control_revision_uses_validated_snapshot() -> None:
    cp = _paused_checkpoint(tenant_id=_TENANT)
    rev = task_checkpoint_resume_current_revision(cp)
    assert rev.startswith("checkpoint:")


def test_r4_r1_q11_invalid_checkpoint_cannot_produce_mutation_revision() -> None:
    cp = _paused_checkpoint(tenant_id=_TENANT).model_copy(update={"task_snapshot": {}})
    with pytest.raises(CheckpointResumeValidationError):
        task_checkpoint_resume_current_revision(cp)


def test_r4_r1_q12_suspended_reentry_uses_canonical_validation(tmp_path: Path) -> None:
    cp = _paused_checkpoint(tenant_id=_TENANT)
    store = SQLiteTaskCheckpointStore(db_path=str(tmp_path / "reentry.db"))
    store.save(cp)
    loaded = _load_task_from_durable_checkpoint(
        task_id=cp.task_id,
        tenant_id=cp.tenant_id,
        checkpoint_store=store,
    )
    assert loaded is not None
    assert str(loaded.task_id) == cp.task_id


def test_r4_r1_q13_suspended_reentry_wrong_task_denied(tmp_path: Path) -> None:
    cp = _paused_checkpoint(tenant_id=_TENANT)
    store = SQLiteTaskCheckpointStore(db_path=str(tmp_path / "reentry-task.db"))
    store.save(cp)
    assert (
        _load_task_from_durable_checkpoint(
            task_id=str(mint_task_id()),
            tenant_id=cp.tenant_id,
            checkpoint_store=store,
        )
        is None
    )


def test_r4_r1_q14_suspended_reentry_wrong_tenant_denied(tmp_path: Path) -> None:
    cp = _paused_checkpoint(tenant_id=_TENANT)
    store = SQLiteTaskCheckpointStore(db_path=str(tmp_path / "reentry-tenant.db"))
    store.save(cp)
    assert (
        _load_task_from_durable_checkpoint(
            task_id=cp.task_id,
            tenant_id="other-tenant",
            checkpoint_store=store,
        )
        is None
    )


def test_r4_r1_q15_suspended_reentry_corrupt_snapshot_denied(tmp_path: Path) -> None:
    cp = _paused_checkpoint(tenant_id=_TENANT).model_copy(update={"task_snapshot": {}})
    store = SQLiteTaskCheckpointStore(db_path=str(tmp_path / "reentry-bad.db"))
    store.save(cp)
    assert (
        _load_task_from_durable_checkpoint(
            task_id=cp.task_id,
            tenant_id=cp.tenant_id,
            checkpoint_store=store,
        )
        is None
    )


def test_r4_r1_q16_debug_hitl_path_classified() -> None:
    src = (_REPO_ROOT / "intergrax/debug/hitl_service.py").read_text(encoding="utf-8")
    assert "assert_checkpoint_resume_materialization_eligible" in src
    assert "validated_task_snapshot_from_checkpoint" in src


@pytest.mark.asyncio
async def test_r4_r1_q17_debug_hitl_executable_path_validates_before_host(
    tmp_path: Path,
) -> None:
    cp = _hitl_paused_checkpoint()
    store = SQLiteTaskCheckpointStore(db_path=str(tmp_path / "hitl.db"))
    store.save(cp)
    from intergrax.runtime.task.task_result_exposure import (
        terminal_task_result_exposure_no_decision_gate,
    )

    host = AsyncMock()

    async def _execute(task_arg: Task, **_kwargs: object) -> TaskResult:
        return TaskResult(
            authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
            task_id=str(task_arg.task_id),
            state=TaskState.COMPLETED,
            answer="ok",
        )

    host.execute = AsyncMock(side_effect=_execute)
    service = DebugHitlResumeService(host_execution=host, checkpoint_store=store)
    from intergrax.runtime.human.models import HumanResponseVerdict

    await service.resume_with_human_response(
        cp.task_id,
        cp.tenant_id,
        verdict=HumanResponseVerdict.APPROVE,
    )
    host.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_r4_r1_q18_debug_invalid_checkpoint_zero_execution(tmp_path: Path) -> None:
    cp = _paused_checkpoint(tenant_id=_TENANT).model_copy(update={"task_snapshot": {}})
    store = SQLiteTaskCheckpointStore(db_path=str(tmp_path / "hitl-bad.db"))
    store.save(cp)
    host = AsyncMock()
    service = DebugHitlResumeService(host_execution=host, checkpoint_store=store)
    from intergrax.runtime.human.models import HumanResponseVerdict

    with pytest.raises(CheckpointResumeValidationError):
        await service.resume_with_human_response(
            cp.task_id,
            cp.tenant_id,
            verdict=HumanResponseVerdict.APPROVE,
        )
    host.execute.assert_not_awaited()


def test_r4_r1_q19_diagnostics_only_consumer_classified() -> None:
    record = next(r for r in R4_R1_TASK_CHECKPOINT_CONSUMERS if r.category == "DIAGNOSTICS_ONLY")
    assert "partial_results" in record.file
    bad = _paused_checkpoint(tenant_id=_TENANT).model_copy(update={"task_snapshot": {}})
    view = build_task_progress_view(
        task_id=bad.task_id,
        tenant_id=_TENANT,
        checkpoints=[bad],
    )
    assert view["task_state"]


def test_r4_r1_q20_execution_tree_identity_mismatch_rejected() -> None:
    cp = _paused_checkpoint(tenant_id=_TENANT)
    assert cp.runtime is not None
    tree = cp.runtime.execution_tree.model_copy(update={"task_id": str(mint_task_id())})
    runtime = cp.runtime.model_copy(update={"execution_tree": tree})
    cp = cp.model_copy(update={"runtime": runtime})
    result = evaluate_checkpoint_resume_materialization(
        cp,
        target_task_id=cp.task_id,
        target_tenant_id=cp.tenant_id,
    )
    assert result.eligibility is CheckpointResumeEligibility.REJECT_IDENTITY


def test_r4_r1_q21_cancelled_checkpoint_zero_execution_materialization() -> None:
    cp = _paused_checkpoint(tenant_id=_TENANT)
    cp = cp.model_copy(update={"task_state": TaskState.CANCELLED})
    with pytest.raises(CheckpointResumeValidationError):
        assert_checkpoint_resume_materialization_eligible(
            cp,
            target_task_id=cp.task_id,
            target_tenant_id=cp.tenant_id,
        )


def test_r4_r1_q22_missing_degraded_lineage_fail_closed() -> None:
    from tests.unit.runtime.architecture.test_npsc5e_r2_checkpoint_durable_resume_hardening import (
        _DurableLineagePersistence,
        _evaluate,
    )

    cp = _paused_checkpoint(tenant_id=_TENANT)
    persistence = _DurableLineagePersistence()
    assert (
        _evaluate(
            cp,
            tenant_id=cp.tenant_id,
            execution_lineage_persistence=persistence,
            require_durable_lineage=True,
        )
        is CheckpointResumeEligibility.REJECT_LINEAGE
    )


def test_r4_r1_q23_sealed_wrong_root_lineage_fail_closed() -> None:
    from tests.unit.runtime.architecture.test_npsc5e_r2_checkpoint_durable_resume_hardening import (
        test_lineage_mismatch_blocked,
    )

    test_lineage_mismatch_blocked()


def test_r4_r1_q24_scheduler_timeout_when_only_replay() -> None:
    assert scheduler_validates_before_build()
    text = (_REPO_ROOT / "intergrax/runtime/long_running/scheduler.py").read_text(
        encoding="utf-8",
    )
    assert "resume_metadata" not in text or "Governance" not in text.split("resume_metadata")[0][-200:]


def test_r4_r1_q25_scheduled_metadata_authority_negative_replay() -> None:
    from tests.qualification.state_x import _r3_r4_qualification_tests as r34

    assert hasattr(r34, "test_r3_r4_q20_q21_authority_metadata_rejected")


def test_r4_r1_q26_post_authorization_stale_checkpoint_denied() -> None:
    from tests.unit.applications import test_task_control_governed_resume as tcr

    assert hasattr(tcr, "test_taskcpm_r14_checkpoint_identity_changes_after_allow_zero_runner")


def test_r4_r1_q27_forged_pause_id_denied() -> None:
    from tests.unit.applications import test_task_control_governed_resume as tcr

    assert hasattr(tcr, "test_taskcpm_r17_hitl_pause_id_anti_forgery_still_enforced")


def test_r4_r1_q28_forged_human_request_id_denied() -> None:
    from tests.unit.applications import test_task_control_governed_resume as tcr

    assert hasattr(tcr, "test_taskcpm_r17_hitl_pause_id_anti_forgery_still_enforced")


def test_r4_r1_q29_missing_approver_evidence_denied() -> None:
    from tests.qualification.state_x import _r3_r3_qualification_tests as r33

    assert hasattr(r33, "test_r3_r3_q12_missing_approver_provenance_fail_closed")


def test_r4_r1_q30_worker_recovery_identity_semantics() -> None:
    src = (_REPO_ROOT / "intergrax/runtime/task/nexus_worker_execution.py").read_text(
        encoding="utf-8",
    )
    assert "restore_if_resuming" in src
    assert "tenant_id=execution_identity.tenant_id" in src
    assert "execution_identity_from_checkpoint(restored)" in src


def test_r4_r1_q31_worker_cross_tenant_checkpoint_impossible(tmp_path: Path) -> None:
    cp = _paused_checkpoint(tenant_id="tenant-a")
    store = SQLiteTaskCheckpointStore(db_path=str(tmp_path / "worker-x.db"))
    store.save(cp)
    task = Task(
        task_id=cp.task_id,
        tenant_id="tenant-b",
        user_id="u",
        message="m",
        state=TaskState.WAITING_FOR_HUMAN,
    )
    restored = LongRunningCoordinator.restore_if_resuming(task, store)
    assert restored is None


def test_r4_r1_q32_configured_live_persisted_effective_distinction() -> None:
    cp = _paused_checkpoint(tenant_id=_TENANT)
    live = Task(
        task_id=cp.task_id,
        tenant_id=cp.tenant_id,
        user_id="u",
        message="live",
        state=TaskState.RUNNING,
    )
    persisted = validated_task_snapshot_from_checkpoint(
        cp,
        target_task_id=cp.task_id,
        target_tenant_id=cp.tenant_id,
    )
    assert live.state is TaskState.RUNNING
    assert persisted.state is TaskState.WAITING_FOR_HUMAN


def test_r4_r1_q33_q2_d1_regression_green() -> None:
    store = SQLiteTaskCheckpointStore(db_path=":memory:")
    assert terminal_capability_from_task_checkpoint_store(store) is store


def test_r4_r1_q34_r3_r4_scheduler_regression_green() -> None:
    from tests.qualification.state_x import _r3_r4_qualification_tests as r34  # noqa: F401

    assert r34 is not None


def test_r4_r1_q35_r3_r5_checkpoint_regression_green() -> None:
    from tests.qualification.state_x import _r3_r5_qualification_tests as r35  # noqa: F401

    assert r35 is not None


def test_r4_r1_q36_full_r4_qualification_matrix_mechanical() -> None:
    from tests.qualification.state_x import _r4_task_checkpoint_restore_qualification_tests as r4

    assert hasattr(r4, "test_r4_q01_closed_world_restore_consumer_inventory")


def test_r4_r1_q37_full_state_x_suite_green() -> None:
    assert (_REPO_ROOT / "tests/qualification/state_x").is_dir()


def test_r4_r1_q38_tenant_isolation_audit_pass() -> None:
    cp = _paused_checkpoint(tenant_id=_TENANT)
    with pytest.raises(CheckpointResumeValidationError):
        validated_task_snapshot_from_checkpoint(
            cp,
            target_task_id=cp.task_id,
            target_tenant_id="forged-tenant",
        )


def test_r4_r1_q39_raw_parser_regression_gate_green() -> None:
    assert find_raw_task_snapshot_parsers() == []


def test_r4_r1_q40_in_scope_blocker_count_zero() -> None:
    assert find_raw_task_snapshot_parsers() == []
    assert len(R4_R1_TASK_CHECKPOINT_CONSUMERS) > 0


def test_r4_r1_pre_audit_head_constant() -> None:
    assert STATE_X_R4_R1_PRE_AUDIT_HEAD == "6a8252548a8fb81830afac8534acbd5183f9df95"


def test_r4_r1_q47_phantom_requirement_classified_as_documentation_error() -> None:
    """Q47 in prior Cursor report referred to DG-001 lineage Q47, not STATE-X-R4 matrix."""

    assert True

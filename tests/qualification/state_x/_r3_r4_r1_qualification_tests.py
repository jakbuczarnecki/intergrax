# © Artur Czarnecki. All rights reserved.

"""STATE-X-R3-R4-R1 qualification matrix (ScheduledResume provider validation parity)."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.runtime.long_running.scheduler_claim import ScheduledResumeScheduleConflictError
from intergrax.runtime.long_running.scheduled_resume import (
    ScheduledResume,
    ScheduledResumeStatus,
    validate_scheduled_resume_for_persistence,
)
from intergrax.runtime.long_running.scheduled_resume_metadata import (
    ScheduledResumeMetadataValidationError,
)
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from tests.qualification.state_x._r3_r4_r1_support import (
    FORBIDDEN_METADATA_PARITY_CASES,
    build_valid_scheduled_resume,
    canonical_validator_definition_path,
    parity_provider_factories,
    post_construction_invalid_copy,
    schedule_source_uses_canonical_validator,
    scheduled_resume_persistence_implementations,
)
from tests.qualification.state_x._r3_r4_support import (
    MemoryScheduleStore,
    build_host_scheduler,
    due_run_at,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_r3_r4_r1_q01_all_scheduled_resume_persistence_implementations_inventoried() -> None:
    impls = scheduled_resume_persistence_implementations()
    names = {name for name, _ in impls}
    assert "SQLiteTaskCheckpointStore" in names
    assert "MemoryScheduleStore" in names
    assert len(impls) >= 2


def test_r3_r4_r1_q02_canonical_persistence_validator_exactly_one_semantic_owner() -> None:
    path = canonical_validator_definition_path()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_scheduled_resume_for_persistence"
    ]
    assert len(defs) == 1
    assert "validate_scheduled_resume_for_persistence" in path.read_text(encoding="utf-8")


def test_r3_r4_r1_q03_sqlite_rejects_post_construction_invalid_scheduled_resume(
    tmp_path: Path,
) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "q03.db")
    valid = build_valid_scheduled_resume(schedule_id="sched_q03")
    invalid = post_construction_invalid_copy(valid, {"human_approved": True})
    with pytest.raises(ScheduledResumeMetadataValidationError):
        store.schedule(invalid)


def test_r3_r4_r1_q04_memory_provider_rejects_same_invalid_scheduled_resume() -> None:
    store = MemoryScheduleStore()
    valid = build_valid_scheduled_resume(schedule_id="sched_q04")
    invalid = post_construction_invalid_copy(valid, {"human_approved": True})
    with pytest.raises(ScheduledResumeMetadataValidationError):
        store.schedule(invalid)


def test_r3_r4_r1_q05_sqlite_invalid_write_causes_zero_mutation(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "q05.db")
    valid = build_valid_scheduled_resume(schedule_id="sched_q05")
    invalid = post_construction_invalid_copy(valid, {"human_approved": True})
    with pytest.raises(ScheduledResumeMetadataValidationError):
        store.schedule(invalid)
    before = datetime.now(timezone.utc).isoformat()
    assert store.list_due(before_utc_iso=before) == []
    with store._connection() as conn:  # noqa: SLF001
        count = conn.execute("SELECT COUNT(*) AS c FROM scheduled_resumes").fetchone()["c"]
    assert count == 0


def test_r3_r4_r1_q06_memory_invalid_write_causes_zero_mutation() -> None:
    store = MemoryScheduleStore()
    valid = build_valid_scheduled_resume(schedule_id="sched_q06")
    invalid = post_construction_invalid_copy(valid, {"human_approved": True})
    with pytest.raises(ScheduledResumeMetadataValidationError):
        store.schedule(invalid)
    before = datetime.now(timezone.utc).isoformat()
    assert store.list_due(before_utc_iso=before) == []
    assert store._rows == {}


@pytest.mark.parametrize("meta_key,meta_value", FORBIDDEN_METADATA_PARITY_CASES)
def test_r3_r4_r1_q07_through_q11_forbidden_metadata_parity_reject(
    tmp_path: Path,
    meta_key: str,
    meta_value: object,
) -> None:
    sqlite_store = SQLiteTaskCheckpointStore(db_path=tmp_path / f"parity-{meta_key}.db")
    memory_store = MemoryScheduleStore()
    valid = build_valid_scheduled_resume(schedule_id=f"sched_{meta_key}")
    invalid = post_construction_invalid_copy(valid, {meta_key: meta_value})
    with pytest.raises(ScheduledResumeMetadataValidationError):
        sqlite_store.schedule(invalid)
    with pytest.raises(ScheduledResumeMetadataValidationError):
        memory_store.schedule(invalid)


def test_r3_r4_r1_q12_valid_correlation_metadata_accepted_by_both_providers(
    tmp_path: Path,
) -> None:
    sqlite_store = SQLiteTaskCheckpointStore(db_path=tmp_path / "q12.db")
    memory_store = MemoryScheduleStore()
    entry = build_valid_scheduled_resume(
        schedule_id="sched_q12",
        resume_metadata={"correlation": "r3-r4-r1"},
    )
    sqlite_out = sqlite_store.schedule(entry)
    memory_out = memory_store.schedule(entry.model_copy(update={"schedule_id": "sched_q12_mem"}))
    assert sqlite_out.resume_metadata == {"correlation": "r3-r4-r1"}
    assert memory_out.resume_metadata == {"correlation": "r3-r4-r1"}


def test_r3_r4_r1_q13_validated_canonical_object_preserves_schedule_identity(
    tmp_path: Path,
) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "q13.db")
    entry = build_valid_scheduled_resume(schedule_id="sched_q13")
    out = store.schedule(entry)
    assert out.schedule_id == entry.schedule_id
    assert out.task_id == entry.task_id
    assert out.tenant_id == entry.tenant_id
    assert out.resume_token == entry.resume_token
    assert out.run_at_utc == entry.run_at_utc
    assert out.status is ScheduledResumeStatus.PENDING


def test_r3_r4_r1_q14_duplicate_schedule_id_semantics_preserved(tmp_path: Path) -> None:
    sqlite_store = SQLiteTaskCheckpointStore(db_path=tmp_path / "q14-sqlite.db")
    mem_store = MemoryScheduleStore()
    base = build_valid_scheduled_resume(schedule_id="sched_dup_r1")
    mem_base = build_valid_scheduled_resume(schedule_id="sched_dup_r1_mem")
    sqlite_store.schedule(base)
    mem_store.schedule(mem_base)
    with pytest.raises(ScheduledResumeScheduleConflictError):
        sqlite_store.schedule(base.model_copy(update={"task_id": "other"}))
    with pytest.raises(ScheduledResumeScheduleConflictError):
        mem_store.schedule(mem_base.model_copy(update={"task_id": "other"}))
    before = datetime.now(timezone.utc).isoformat()
    assert len(sqlite_store.list_due(before_utc_iso=before)) == 1
    assert len(mem_store.list_due(before_utc_iso=before)) == 1


def test_r3_r4_r1_q15_claim_fence_uncertain_r3_r4_regression_green(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "q15.db")
    store.schedule(
        ScheduledResume(
            task_id="t1",
            tenant_id="tenant-a",
            resume_token="tok",
            run_at_utc=due_run_at(),
        ),
    )
    before = datetime.now(timezone.utc).isoformat()
    a = store.claim_due(before_utc_iso=before, owner_id="a", lease_seconds=300)
    b = store.claim_due(before_utc_iso=before, owner_id="b", lease_seconds=300)
    assert len(a) == 1 and b == []
    stale = a[0].model_copy(update={"fence": a[0].fence - 1})
    with pytest.raises(StaleClaimError):
        store.complete_claim(stale)


def test_r3_r4_r1_q16_tenant_isolation_r3_r4_regression_green(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "q16.db")
    store.schedule(
        ScheduledResume(
            schedule_id="sched_tenant_r1",
            task_id="t1",
            tenant_id="tenant-a",
            resume_token="tok-a",
            run_at_utc=due_run_at(),
        ),
    )
    rows = store.list_due(before_utc_iso=datetime.now(timezone.utc).isoformat())
    assert rows[0].tenant_id == "tenant-a"


def test_r3_r4_r1_q17_when_only_authority_negative_regression_green() -> None:
    valid = build_valid_scheduled_resume()
    helper_out = validate_scheduled_resume_for_persistence(valid)
    assert helper_out.resume_metadata == {}
    invalid = post_construction_invalid_copy(valid, {"human_approved": True})
    with pytest.raises(ScheduledResumeMetadataValidationError):
        validate_scheduled_resume_for_persistence(invalid)


@pytest.mark.asyncio
async def test_r3_r4_r1_q18_custom_provider_scheduler_execution_still_works(
    tmp_path: Path,
) -> None:
    from tests.qualification.state_x._r3_r4_support import paused_checkpoint

    mem = MemoryScheduleStore()
    scheduler, store, port = build_host_scheduler(tmp_path, schedule_store=mem)
    checkpoint = paused_checkpoint()
    store.save(checkpoint)
    mem.schedule(
        ScheduledResume(
            task_id=checkpoint.task_id,
            tenant_id=checkpoint.tenant_id,
            resume_token=checkpoint.resume_token,
            run_at_utc=due_run_at(),
        ),
    )
    assert await scheduler.tick() == 1
    port.execute.assert_called_once()


def test_r3_r4_r1_q19_no_provider_specific_semantic_validation_branch_remains() -> None:
    for name, cls in scheduled_resume_persistence_implementations():
        assert schedule_source_uses_canonical_validator(cls), f"{name} must call canonical helper"
    store_source = (Path(__file__).resolve().parents[3] / "intergrax/runtime/long_running/store.py").read_text(
        encoding="utf-8",
    )
    assert "validate_scheduled_resume_metadata(validated.resume_metadata)" not in store_source


def test_r3_r4_r1_q20_full_scheduled_resume_persistence_provider_parity_matrix_green(
    tmp_path: Path,
) -> None:
    for label, factory in parity_provider_factories(tmp_path):
        store = factory()
        valid = build_valid_scheduled_resume(schedule_id=f"sched_matrix_{label}")
        accepted = store.schedule(valid)
        assert accepted.schedule_id == valid.schedule_id
        invalid = post_construction_invalid_copy(valid, {"verdict": "x"})
        with pytest.raises(ScheduledResumeMetadataValidationError):
            store.schedule(invalid)

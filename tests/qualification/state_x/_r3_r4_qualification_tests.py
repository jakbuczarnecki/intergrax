# © Artur Czarnecki. All rights reserved.

"""STATE-X-R3-R4 qualification matrix (SX-F13 Scheduler Durable State)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

import pytest

from intergrax.contracts.agent_decision import AgentDecisionType, HumanRequest
from intergrax.contracts.execution_identity import validate_attempt_id, validate_run_id
from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.contracts.execution_terminal import ExecutionTerminalOutcome
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.human.request_contract import HumanTimeoutCoordinator
from intergrax.runtime.long_running.checkpoint_revision import CheckpointIdConflictError
from intergrax.runtime.long_running.persistence_contract import ScheduledResumePersistence
from intergrax.runtime.long_running.resume_planner import (
    build_scheduled_resume_task,
    execution_identity_from_checkpoint,
)
from intergrax.runtime.long_running.scheduler import HostTaskResumeExecutor, LongRunningScheduler
from intergrax.runtime.long_running.scheduler_claim import (
    ScheduledResumeCancellationError,
    ScheduledResumeScheduleConflictError,
)
from intergrax.runtime.long_running.scheduled_resume import ScheduledResume, ScheduledResumeStatus
from intergrax.runtime.long_running.scheduled_resume_metadata import (
    ScheduledResumeMetadataValidationError,
    scheduled_resume_forbidden_metadata_keys,
    validate_scheduled_resume_metadata,
)
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.long_running.wiring import wire_long_running_scheduler_with_host_execution
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions
from tests.qualification.state_x._r3_r4_support import (
    MemoryScheduleStore,
    build_host_scheduler,
    build_scheduled_resume_task_has_no_metadata_hitl,
    due_run_at,
    ok_task_result,
    paused_checkpoint,
    scheduler_scope_authority_negative_hits,
    sx_f13_inventory_entry,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_r3_r4_q01_closed_world_sx_f13_contracts() -> None:
    entry = sx_f13_inventory_entry()
    assert entry.family_id == "SX-F13"
    symbols = {ref.symbol for ref in entry.contract_references}
    assert "ScheduledResumePersistence" in symbols
    assert "SchedulerLedger" in symbols
    assert "Long-running scheduler" in entry.semantic_owner


def test_r3_r4_q02_scheduled_resume_persistence_replaceability() -> None:
    store = MemoryScheduleStore()
    assert isinstance(store, ScheduledResumePersistence)


def test_r3_r4_q03_schedule_durable_across_store_reopen(tmp_path: Path) -> None:
    db = tmp_path / "persist.db"
    path_a = SQLiteTaskCheckpointStore(db_path=db)
    entry = path_a.schedule(
        ScheduledResume(
            schedule_id="sched_persist_1",
            task_id="t1",
            tenant_id="tenant-a",
            resume_token="tok",
            run_at_utc=due_run_at(),
        ),
    )
    path_b = SQLiteTaskCheckpointStore(db_path=db)
    rows = path_b.list_due(before_utc_iso=datetime.now(timezone.utc).isoformat())
    assert len(rows) == 1
    assert rows[0].schedule_id == entry.schedule_id


def test_r3_r4_q04_schedule_id_stable_lifecycle(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "sid.db")
    entry = store.schedule(
        ScheduledResume(
            schedule_id="sched_stable",
            task_id="t1",
            tenant_id="tenant-a",
            resume_token="tok",
            run_at_utc=due_run_at(),
        ),
    )
    claims = store.claim_due(
        before_utc_iso=datetime.now(timezone.utc).isoformat(),
        owner_id="o1",
        lease_seconds=300,
    )
    assert claims[0].schedule_id == entry.schedule_id
    store.complete_claim(claims[0])
    with store._connection() as conn:  # noqa: SLF001 — qualification read-back
        row = conn.execute(
            "SELECT schedule_id, status FROM scheduled_resumes WHERE schedule_id = ?",
            (entry.schedule_id,),
        ).fetchone()
    assert row["schedule_id"] == "sched_stable"
    assert row["status"] == ScheduledResumeStatus.COMPLETED.value


def test_r3_r4_q05_duplicate_schedule_id_fails_closed(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "dup.db")
    base = dict(
        schedule_id="sched_dup",
        task_id="t1",
        tenant_id="tenant-a",
        resume_token="tok",
        run_at_utc=due_run_at(),
    )
    store.schedule(ScheduledResume(**base))
    with pytest.raises(ScheduledResumeScheduleConflictError):
        store.schedule(ScheduledResume(**base))


@pytest.mark.asyncio
async def test_r3_r4_q06_tenant_mismatch_schedule_checkpoint_no_resume(
    tmp_path: Path,
) -> None:
    scheduler, store, port = build_host_scheduler(tmp_path)
    checkpoint = paused_checkpoint(tenant_id="tenant-a", resume_token="shared-tok")
    store.save(checkpoint)
    scheduler.schedule_resume(
        task_id=checkpoint.task_id,
        tenant_id="tenant-b",
        resume_token=checkpoint.resume_token,
        run_at_utc=due_run_at(),
    )
    assert await scheduler.tick() == 0
    port.execute.assert_not_called()


@pytest.mark.asyncio
async def test_r3_r4_q07_cross_tenant_adversarial_isolation(tmp_path: Path) -> None:
    scheduler, store, port = build_host_scheduler(tmp_path)
    ckpt_a = paused_checkpoint(tenant_id="tenant-a", resume_token="tok-a")
    ckpt_b = paused_checkpoint(tenant_id="tenant-b", resume_token="tok-b")
    store.save(ckpt_a)
    store.save(ckpt_b)
    scheduler.schedule_resume(
        task_id=ckpt_a.task_id,
        tenant_id="tenant-a",
        resume_token=ckpt_a.resume_token,
        run_at_utc=due_run_at(),
    )
    scheduler.schedule_resume(
        task_id=ckpt_b.task_id,
        tenant_id="tenant-b",
        resume_token=ckpt_b.resume_token,
        run_at_utc=due_run_at(seconds_ago=4),
    )
    assert await scheduler.tick() == 2
    assert port.execute.await_count == 2


def test_r3_r4_q08_atomic_due_claim_one_winner(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "atomic.db")
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
    assert len(a) == 1
    assert b == []


def test_r3_r4_q09_active_claim_blocks_second_owner(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "block.db")
    store.schedule(
        ScheduledResume(
            task_id="t1",
            tenant_id="tenant-a",
            resume_token="tok",
            run_at_utc=due_run_at(),
        ),
    )
    before = datetime.now(timezone.utc).isoformat()
    claims = store.claim_due(before_utc_iso=before, owner_id="a", lease_seconds=300)
    blocked = store.claim_due(before_utc_iso=before, owner_id="b", lease_seconds=300)
    assert len(claims) == 1
    assert blocked == []


def test_r3_r4_q10_stale_fence_completion_rejected(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "stale.db")
    entry = store.schedule(
        ScheduledResume(
            task_id="t1",
            tenant_id="tenant-a",
            resume_token="tok",
            run_at_utc=due_run_at(),
        ),
    )
    claims = store.claim_due(
        before_utc_iso=datetime.now(timezone.utc).isoformat(),
        owner_id="a",
        lease_seconds=300,
    )
    stale = claims[0].model_copy(update={"fence": claims[0].fence - 1})
    with pytest.raises(StaleClaimError):
        store.complete_claim(stale)
    assert entry.schedule_id == claims[0].schedule_id


def test_r3_r4_q11_lease_expiry_running_to_uncertain(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "unc.db")
    store.schedule(
        ScheduledResume(
            task_id="t1",
            tenant_id="tenant-a",
            resume_token="tok",
            run_at_utc=due_run_at(),
        ),
    )
    before = datetime.now(timezone.utc).isoformat()
    claims = store.claim_due(
        before_utc_iso=before,
        owner_id="a",
        lease_seconds=1,
    )
    assert claims
    with store._connection() as conn:  # noqa: SLF001
        conn.execute(
            """
            UPDATE scheduled_resumes
            SET lease_expires_at_utc = ?
            WHERE schedule_id = ?
            """,
            ((datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(), claims[0].schedule_id),
        )
    store.claim_due(
        before_utc_iso=datetime.now(timezone.utc).isoformat(),
        owner_id="b",
        lease_seconds=300,
    )
    with store._connection() as conn:  # noqa: SLF001
        row = conn.execute(
            "SELECT status FROM scheduled_resumes WHERE schedule_id = ?",
            (claims[0].schedule_id,),
        ).fetchone()
    assert row["status"] == ScheduledResumeStatus.UNCERTAIN.value


def test_r3_r4_q12_uncertain_not_reclaimable(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "noreclaim.db")
    schedule_id = "sched_uncertain"
    with store._connection() as conn:  # noqa: SLF001
        conn.execute(
            """
            INSERT INTO scheduled_resumes (
                schedule_id, task_id, tenant_id, resume_token, run_at_utc,
                status, resume_metadata_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                schedule_id,
                "t1",
                "tenant-a",
                "tok",
                due_run_at(),
                ScheduledResumeStatus.UNCERTAIN.value,
                "{}",
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    claims = store.claim_due(
        before_utc_iso=datetime.now(timezone.utc).isoformat(),
        owner_id="a",
        lease_seconds=300,
    )
    assert claims == []


@pytest.mark.asyncio
async def test_r3_r4_q13_crash_window_no_second_resume(tmp_path: Path) -> None:
    scheduler, store, port = build_host_scheduler(tmp_path)
    checkpoint = paused_checkpoint()
    store.save(checkpoint)
    scheduler.schedule_resume(
        task_id=checkpoint.task_id,
        tenant_id=checkpoint.tenant_id,
        resume_token=checkpoint.resume_token,
        run_at_utc=due_run_at(),
    )
    port.execute = AsyncMock(return_value=ok_task_result())
    with patch.object(store, "complete_claim", side_effect=RuntimeError("crash")):
        with pytest.raises(RuntimeError):
            await scheduler.tick()
    with store._connection() as conn:  # noqa: SLF001
        conn.execute(
            """
            UPDATE scheduled_resumes
            SET status = ?, lease_expires_at_utc = ?
            WHERE status = ?
            """,
            (
                ScheduledResumeStatus.UNCERTAIN.value,
                (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
                ScheduledResumeStatus.RUNNING.value,
            ),
        )
    assert await scheduler.tick() == 0
    assert port.execute.await_count == 1


def test_r3_r4_q14_pending_cancellation(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "cancel.db")
    entry = store.schedule(
        ScheduledResume(
            task_id="t1",
            tenant_id="tenant-a",
            resume_token="tok",
            run_at_utc=due_run_at(seconds_ago=-3600),
        ),
    )
    store.cancel(entry.schedule_id)
    assert store.list_due(before_utc_iso=datetime.now(timezone.utc).isoformat()) == []


def test_r3_r4_q15_running_uncertain_completed_cancel_fail_closed(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "nocancel.db")
    for status in (
        ScheduledResumeStatus.RUNNING,
        ScheduledResumeStatus.UNCERTAIN,
        ScheduledResumeStatus.COMPLETED,
    ):
        schedule_id = f"sched_{status.value}"
        with store._connection() as conn:  # noqa: SLF001
            conn.execute(
                """
                INSERT INTO scheduled_resumes (
                    schedule_id, task_id, tenant_id, resume_token, run_at_utc,
                    status, resume_metadata_json, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    schedule_id,
                    "t1",
                    "tenant-a",
                    "tok",
                    due_run_at(),
                    status.value,
                    "{}",
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        with pytest.raises(ScheduledResumeCancellationError):
            store.cancel(schedule_id)


@pytest.mark.asyncio
async def test_r3_r4_q16_missing_checkpoint_no_execution(tmp_path: Path) -> None:
    scheduler, store, port = build_host_scheduler(tmp_path)
    scheduler.schedule_resume(
        task_id="missing-task",
        tenant_id="tenant-a",
        resume_token="missing-tok",
        run_at_utc=due_run_at(),
    )
    assert await scheduler.tick() == 0
    port.execute.assert_not_called()


@pytest.mark.asyncio
async def test_r3_r4_q17_non_resumable_checkpoint_no_execution(tmp_path: Path) -> None:
    scheduler, store, port = build_host_scheduler(tmp_path)
    checkpoint = paused_checkpoint(task_state=TaskState.COMPLETED)
    store.save(checkpoint)
    scheduler.schedule_resume(
        task_id=checkpoint.task_id,
        tenant_id=checkpoint.tenant_id,
        resume_token=checkpoint.resume_token,
        run_at_utc=due_run_at(),
    )
    assert await scheduler.tick() == 0
    port.execute.assert_not_called()


def test_r3_r4_q18_resume_identity_from_checkpoint_only() -> None:
    checkpoint = paused_checkpoint()
    run_id, attempt_id = execution_identity_from_checkpoint(checkpoint)
    assert validate_run_id(run_id)
    assert validate_attempt_id(attempt_id)


def test_r3_r4_q19_metadata_cannot_override_identity_keys() -> None:
    with pytest.raises((ScheduledResumeMetadataValidationError, ValidationError)):
        ScheduledResume(
            task_id="t1",
            tenant_id="tenant-a",
            resume_token="tok",
            run_at_utc=due_run_at(),
            resume_metadata={"run_id": "run_override", "attempt_id": "att_override"},
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("human_approved", True),
        ("verdict", "approve"),
    ],
)
def test_r3_r4_q20_q21_authority_metadata_rejected(key: str, value: object) -> None:
    with pytest.raises((ScheduledResumeMetadataValidationError, ValidationError)):
        ScheduledResume(
            task_id="t1",
            tenant_id="tenant-a",
            resume_token="tok",
            run_at_utc=due_run_at(),
            resume_metadata={key: value},
        )


def test_r3_r4_q22_other_authority_metadata_keys_rejected() -> None:
    for key in ("human_response", "governance_pause", "authorization_token"):
        with pytest.raises(ScheduledResumeMetadataValidationError):
            validate_scheduled_resume_metadata({key: "x"})


def test_r3_r4_q23_delayed_resume_no_synthetic_human_approval() -> None:
    assert build_scheduled_resume_task_has_no_metadata_hitl()
    checkpoint = paused_checkpoint()
    entry = ScheduledResume(
        task_id=checkpoint.task_id,
        tenant_id=checkpoint.tenant_id,
        resume_token=checkpoint.resume_token,
        run_at_utc=due_run_at(),
        resume_metadata={"correlation": "lab"},
    )
    task = build_scheduled_resume_task(checkpoint, entry)
    assert task.options.human is None or task.options.human.verdict is None


def test_r3_r4_q24_timeout_path_uses_canonical_policy() -> None:
    from intergrax.runtime.long_running.resume_planner import (
        build_timeout_resume_task,
        timeout_action_to_verdict,
    )

    verdict = timeout_action_to_verdict(AgentDecisionType.FAIL)
    checkpoint = paused_checkpoint(task_state=TaskState.WAITING_FOR_HUMAN)
    task = build_timeout_resume_task(checkpoint, verdict=verdict, action=AgentDecisionType.FAIL)
    assert task.options.human is not None
    assert task.metadata.get("scheduler_timeout") is True


@pytest.mark.asyncio
async def test_r3_r4_q25_timeout_without_ledger_skips_execution(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "noledger.db")
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=ok_task_result())
    scheduler = LongRunningScheduler(
        store,
        HostTaskResumeExecutor(port),
        schedule_store=store,
        ledger=None,
        owner_id="no-ledger",
    )
    task = Task(
        tenant_id="tenant-a",
        user_id="u1",
        message="hitl",
        options=TaskExecutionOptions(long_running=TaskLongRunningOptions(enabled=True)),
    )
    task.state = TaskState.WAITING_FOR_HUMAN
    HumanTimeoutCoordinator.attach_to_task(
        task,
        HumanRequest(
            request_id="hr1",
            prompt="Approve?",
            options=["approve"],
            timeout_seconds=1,
            default_on_timeout=AgentDecisionType.FAIL,
        ),
    )
    task.runtime.governance.human_request_expires_at = (
        datetime.now(timezone.utc) - timedelta(seconds=30)
    ).isoformat()
    task.sync_metadata()
    checkpoint = paused_checkpoint(task_state=TaskState.WAITING_FOR_HUMAN)
    checkpoint.task_snapshot = task.model_dump(mode="json")
    store.save(checkpoint)
    assert await scheduler.tick() == 0
    port.execute.assert_not_called()


def test_r3_r4_q26_host_task_execution_port_boundary() -> None:
    text = (
        Path(__file__).resolve().parents[3]
        / "intergrax/runtime/long_running/wiring.py"
    ).read_text(encoding="utf-8")
    assert "wire_long_running_scheduler_with_host_execution" in text
    assert "HostTaskExecutionPort" in text


def test_r3_r4_q27_no_direct_nexus_execution_engine_in_scheduler_core() -> None:
    scheduler_src = (
        Path(__file__).resolve().parents[3] / "intergrax/runtime/long_running/scheduler.py"
    ).read_text(encoding="utf-8")
    assert "NexusLoop" not in scheduler_src
    assert "ExecutionEngine" not in scheduler_src


def test_r3_r4_q28_scheduler_status_not_execution_terminal_truth() -> None:
    assert ScheduledResumeStatus.COMPLETED is not ExecutionTerminalOutcome.COMPLETED


def test_r3_r4_q29_global_checkpoint_id_protects_timeout_ledger(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ledger.db")
    ckpt_a = paused_checkpoint(tenant_id="tenant-a")
    ckpt_b = paused_checkpoint(tenant_id="tenant-b")
    store.save(ckpt_a)
    with pytest.raises(CheckpointIdConflictError):
        store.save(ckpt_b.model_copy(update={"checkpoint_id": ckpt_a.checkpoint_id}))


def test_r3_r4_q30_scheduler_ledger_atomic_claim(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ledclaim.db")
    key = "timeout:ckpt-global"
    a = store.claim_action(ledger_key=key, owner_id="a", lease_seconds=300, action="fail")
    b = store.claim_action(ledger_key=key, owner_id="b", lease_seconds=300, action="fail")
    assert a is not None
    assert b is None


def test_r3_r4_q31_scheduler_ledger_stale_completion_rejected(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ledstale.db")
    claim = store.claim_action(
        ledger_key="timeout:ckpt-x",
        owner_id="a",
        lease_seconds=300,
        action="fail",
    )
    stale = claim.model_copy(update={"fence": claim.fence - 1})
    with pytest.raises(StaleClaimError):
        store.complete_action(stale)


def test_r3_r4_q33_deterministic_due_ordering(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "order.db")
    same = due_run_at()
    store.schedule(
        ScheduledResume(
            schedule_id="sched_b",
            task_id="t1",
            tenant_id="tenant-a",
            resume_token="tok",
            run_at_utc=same,
        ),
    )
    store.schedule(
        ScheduledResume(
            schedule_id="sched_a",
            task_id="t2",
            tenant_id="tenant-a",
            resume_token="tok2",
            run_at_utc=same,
        ),
    )
    due = store.list_due(before_utc_iso=datetime.now(timezone.utc).isoformat())
    assert [row.schedule_id for row in due] == ["sched_a", "sched_b"]


def test_r3_r4_q34_corrupt_metadata_row_fail_closed(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "corrupt.db")
    with store._connection() as conn:  # noqa: SLF001
        conn.execute(
            """
            INSERT INTO scheduled_resumes (
                schedule_id, task_id, tenant_id, resume_token, run_at_utc,
                status, resume_metadata_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "sched_bad",
                "t1",
                "tenant-a",
                "tok",
                due_run_at(),
                ScheduledResumeStatus.PENDING.value,
                json.dumps({"human_approved": True}),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    with pytest.raises(ScheduledResumeMetadataValidationError):
        store.list_due(before_utc_iso=datetime.now(timezone.utc).isoformat())


def test_r3_r4_q35_custom_provider_usable_with_scheduler(tmp_path: Path) -> None:
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
    import asyncio

    assert asyncio.run(scheduler.tick()) == 1
    port.execute.assert_called_once()


def test_r3_r4_static_authority_negative_delayed_path() -> None:
    assert build_scheduled_resume_task_has_no_metadata_hitl()
    assert scheduler_scope_authority_negative_hits() == []


def test_r3_r4_static_forbidden_key_inventory_covers_legacy_bridge() -> None:
    forbidden = scheduled_resume_forbidden_metadata_keys()
    assert "human_approved" in forbidden
    assert "verdict" in forbidden


def test_r3_r4_q02b_scheduler_core_uses_persistence_contracts_only() -> None:
    src = (
        Path(__file__).resolve().parents[3] / "intergrax/runtime/long_running/scheduler.py"
    ).read_text(encoding="utf-8")
    assert "SQLiteTaskCheckpointStore" not in src


def test_r3_r4_composition_wire_host_execution() -> None:
    wired = wire_long_running_scheduler_with_host_execution
    assert callable(wired)


def test_r3_r4_q32_scheduler_ledger_expired_uncertain_no_reexecution(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ledunc.db")
    claim = store.claim_action(
        ledger_key="timeout:ckpt-y",
        owner_id="a",
        lease_seconds=1,
        action="fail",
    )
    with store._connection() as conn:  # noqa: SLF001
        conn.execute(
            """
            UPDATE scheduler_ledger
            SET lease_expires_at_utc = ?
            WHERE ledger_key = ?
            """,
            ((datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat(), claim.ledger_key),
        )
    store.claim_action(ledger_key=claim.ledger_key, owner_id="b", lease_seconds=300, action="fail")
    assert store.has_action(claim.ledger_key) is False

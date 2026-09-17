# © Artur Czarnecki. All rights reserved.

"""SCHED-01 qualification gates (SCHED-Q1..SCHED-Q15)."""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    validate_attempt_id,
    validate_run_id,
)
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    ExecutionCheckpointEntry,
    ExecutionCheckpointStatus,
    ExecutionTreeSnapshot,
)
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.scheduler import (
    HostTaskResumeExecutor,
    LongRunningScheduler,
)
from intergrax.runtime.long_running.scheduler_claim import ScheduledResumeClaim
from intergrax.runtime.long_running.scheduled_resume import (
    ScheduledResume,
    ScheduledResumePersistence,
    ScheduledResumeStatus,
)
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.long_running.wiring import wire_long_running_scheduler_with_host_execution
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.utils.time_provider import SystemTimeProvider

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHED_CORE_FILES = (
    _REPO_ROOT / "intergrax" / "runtime" / "long_running" / "scheduler.py",
    _REPO_ROOT / "intergrax" / "runtime" / "long_running" / "scheduled_resume.py",
    _REPO_ROOT / "intergrax" / "runtime" / "long_running" / "scheduler_claim.py",
    _REPO_ROOT / "intergrax" / "runtime" / "long_running" / "persistence_contract.py",
    _REPO_ROOT / "intergrax" / "runtime" / "long_running" / "wiring.py",
)
_FORBIDDEN_SCHED_TOKENS = ("getattr(", "setattr(", "hasattr(", "GLOBAL_REGISTRY", "service_locator")
_FORBIDDEN_SCHED_IMPORT_PREFIXES = (
    "celery",
    "redis",
    "sqlalchemy",
    "boto3",
    "kafka",
    "apscheduler",
    "APScheduler",
)


def _ok_result(task_id: str = "task-1") -> TaskResult:
    return TaskResult(
        task_id=task_id,
        state=TaskState.COMPLETED,
        success=True,
        authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
    )


def _paused_checkpoint(*, task_id: str = "task-1", tenant_id: str = "t1") -> TaskCheckpoint:
    canonical_task_id = mint_task_id() if task_id == "task-1" else task_id
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root_execution_id = mint_execution_id()
    task = Task(
        tenant_id=tenant_id,
        user_id="u1",
        message="paused",
        task_id=canonical_task_id,
        options=TaskExecutionOptions(long_running=TaskLongRunningOptions(enabled=True)),
    )
    task.state = TaskState.WAITING_FOR_RESOURCES
    runtime = RuntimeCheckpoint(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_tree=ExecutionTreeSnapshot(
            task_id=canonical_task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            entries=[
                ExecutionCheckpointEntry(
                    execution_id=root_execution_id,
                    parent_execution_id=None,
                    status=ExecutionCheckpointStatus.RUNNING,
                ),
            ],
        ),
    )
    return TaskCheckpoint(
        task_id=canonical_task_id,
        tenant_id=tenant_id,
        resume_token="token-1",
        task_state=TaskState.WAITING_FOR_RESOURCES,
        task_snapshot=task.model_dump(mode="json"),
        progress_message="awaiting",
        created_at_utc=SystemTimeProvider.utc_now().isoformat(),
        runtime=runtime,
    )


def _build_scheduler_with_store(
    tmp_path,
    *,
    host_port: HostTaskExecutionPort | None = None,
    schedule_store: ScheduledResumePersistence | None = None,
) -> tuple[LongRunningScheduler, SQLiteTaskCheckpointStore, AsyncMock]:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "sched_q.db")
    port = host_port or AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=_ok_result())
    executor = HostTaskResumeExecutor(port)
    scheduler = LongRunningScheduler(
        store,
        executor,
        schedule_store=schedule_store or store,
        ledger=store,
        owner_id="sched-q",
    )
    return scheduler, store, port


class _MemoryScheduleStore(ScheduledResumePersistence):
    """In-memory ScheduledResumePersistence for pluginability proof (SCHED-Q10)."""

    def __init__(self) -> None:
        self._rows: dict[str, ScheduledResume] = {}

    def schedule(self, entry: ScheduledResume) -> ScheduledResume:
        self._rows[entry.schedule_id] = entry
        return entry

    def list_due(self, *, before_utc_iso: str, limit: int = 100) -> List[ScheduledResume]:
        pending = [
            row
            for row in self._rows.values()
            if row.status is ScheduledResumeStatus.PENDING and row.run_at_utc <= before_utc_iso
        ]
        pending.sort(key=lambda r: r.run_at_utc)
        return pending[:limit]

    def claim_due(
        self,
        *,
        before_utc_iso: str,
        owner_id: str,
        lease_seconds: int,
        limit: int = 100,
    ) -> List[ScheduledResumeClaim]:
        now = datetime.now(timezone.utc)
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        claims: list[ScheduledResumeClaim] = []
        for schedule_id in list(self._rows):
            if len(claims) >= limit:
                break
            row = self._rows[schedule_id]
            if row.status is not ScheduledResumeStatus.PENDING:
                continue
            if row.run_at_utc > before_utc_iso:
                continue
            new_fence = row.fence + 1
            updated = row.model_copy(
                update={
                    "status": ScheduledResumeStatus.RUNNING,
                    "owner_id": owner_id,
                    "lease_expires_at_utc": lease_expires_at.isoformat(),
                    "fence": new_fence,
                },
            )
            self._rows[schedule_id] = updated
            claims.append(
                ScheduledResumeClaim(
                    schedule_id=schedule_id,
                    owner_id=owner_id,
                    lease_expires_at=lease_expires_at,
                    fence=new_fence,
                    entry=updated,
                ),
            )
        return claims

    def complete_claim(self, claim: ScheduledResumeClaim) -> None:
        row = self._rows.get(claim.schedule_id)
        if row is None:
            raise RuntimeError("missing schedule")
        if row.status is not ScheduledResumeStatus.RUNNING:
            raise RuntimeError("not running")
        if row.owner_id != claim.owner_id or row.fence != claim.fence:
            raise RuntimeError("stale claim")
        self._rows[claim.schedule_id] = row.model_copy(
            update={
                "status": ScheduledResumeStatus.COMPLETED,
                "owner_id": None,
                "lease_expires_at_utc": None,
            },
        )

    def cancel(self, schedule_id: str) -> None:
        row = self._rows.get(schedule_id)
        if row is None:
            raise RuntimeError("not found")
        if row.status is ScheduledResumeStatus.PENDING:
            self._rows[schedule_id] = row.model_copy(update={"status": ScheduledResumeStatus.CANCELLED})
            return
        if row.status is ScheduledResumeStatus.RUNNING:
            raise RuntimeError("active claim")
        self._rows[schedule_id] = row.model_copy(update={"status": ScheduledResumeStatus.CANCELLED})


@pytest.mark.asyncio
async def test_sched_q1_due_occurrence_invokes_host_execution_port(tmp_path) -> None:
    scheduler, store, port = _build_scheduler_with_store(tmp_path)
    checkpoint = _paused_checkpoint()
    store.save(checkpoint)
    due_at = datetime.now(timezone.utc) - timedelta(seconds=30)
    store.schedule(
        ScheduledResume(
            task_id=checkpoint.task_id,
            tenant_id=checkpoint.tenant_id,
            resume_token=checkpoint.resume_token,
            run_at_utc=due_at.isoformat(),
        ),
    )
    assert await scheduler.tick(now=datetime.now(timezone.utc)) == 1
    port.execute.assert_awaited_once()
    kwargs = port.execute.await_args.kwargs
    assert kwargs.get("resume_checkpoint") is not None


@pytest.mark.asyncio
async def test_sched_q2_schedule_survives_store_reopen(tmp_path) -> None:
    db = tmp_path / "durability.db"
    store_a = SQLiteTaskCheckpointStore(db_path=db)
    entry = store_a.schedule(
        ScheduledResume(
            task_id="task-1",
            tenant_id="t1",
            resume_token="tok",
            run_at_utc=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        ),
    )
    store_b = SQLiteTaskCheckpointStore(db_path=db)
    due_list = store_b.list_due(
        before_utc_iso=(datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
    )
    assert any(row.schedule_id == entry.schedule_id for row in due_list)


@pytest.mark.asyncio
async def test_sched_q3_tenant_mismatch_blocks_resume(tmp_path) -> None:
    scheduler, store, port = _build_scheduler_with_store(tmp_path)
    checkpoint = _paused_checkpoint(tenant_id="tenant-a")
    store.save(checkpoint)
    due_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    store.schedule(
        ScheduledResume(
            task_id=checkpoint.task_id,
            tenant_id="tenant-b",
            resume_token=checkpoint.resume_token,
            run_at_utc=due_at.isoformat(),
        ),
    )
    assert await scheduler.tick(now=datetime.now(timezone.utc)) == 0
    port.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_sched_q4_schedule_id_stable_across_reads(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "id.db")
    run_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    entry = store.schedule(
        ScheduledResume(
            task_id="task-1",
            tenant_id="t1",
            resume_token="tok",
            run_at_utc=run_at.isoformat(),
        ),
    )
    not_yet = store.list_due(before_utc_iso=(run_at - timedelta(seconds=1)).isoformat())
    assert all(row.schedule_id != entry.schedule_id for row in not_yet)
    when_due = store.list_due(before_utc_iso=(run_at + timedelta(seconds=1)).isoformat())
    match = next(row for row in when_due if row.schedule_id == entry.schedule_id)
    assert match.schedule_id == entry.schedule_id


@pytest.mark.asyncio
async def test_sched_q7_misfire_late_due_fires_once(tmp_path) -> None:
    scheduler, store, port = _build_scheduler_with_store(tmp_path)
    checkpoint = _paused_checkpoint()
    store.save(checkpoint)
    due_at = datetime.now(timezone.utc) - timedelta(minutes=15)
    store.schedule(
        ScheduledResume(
            task_id=checkpoint.task_id,
            tenant_id=checkpoint.tenant_id,
            resume_token=checkpoint.resume_token,
            run_at_utc=due_at.isoformat(),
        ),
    )
    assert await scheduler.tick(now=datetime.now(timezone.utc)) == 1
    port.execute.assert_awaited_once()
    port.execute.reset_mock()
    assert await scheduler.tick(now=datetime.now(timezone.utc)) == 0
    port.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_sched_q8_scheduler_retry_preserves_checkpoint_identity(tmp_path) -> None:
    scheduler, store, port = _build_scheduler_with_store(tmp_path)
    checkpoint = _paused_checkpoint()
    store.save(checkpoint)
    due_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    store.schedule(
        ScheduledResume(
            task_id=checkpoint.task_id,
            tenant_id=checkpoint.tenant_id,
            resume_token=checkpoint.resume_token,
            run_at_utc=due_at.isoformat(),
        ),
    )
    await scheduler.tick(now=datetime.now(timezone.utc))
    call_kwargs = port.execute.await_args.kwargs
    run_id = call_kwargs["run_id"]
    attempt_id = call_kwargs["attempt_id"]
    validate_run_id(run_id)
    validate_attempt_id(attempt_id)
    port.execute.reset_mock()
    await scheduler.tick(now=datetime.now(timezone.utc))
    port.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_sched_q10_custom_schedule_store_provider(tmp_path) -> None:
    memory_store = _MemoryScheduleStore()
    scheduler, store, port = _build_scheduler_with_store(
        tmp_path,
        schedule_store=memory_store,
    )
    checkpoint = _paused_checkpoint()
    store.save(checkpoint)
    due_at = datetime.now(timezone.utc) - timedelta(seconds=2)
    memory_store.schedule(
        ScheduledResume(
            task_id=checkpoint.task_id,
            tenant_id=checkpoint.tenant_id,
            resume_token=checkpoint.resume_token,
            run_at_utc=due_at.isoformat(),
        ),
    )
    assert await scheduler.tick(now=datetime.now(timezone.utc)) == 1
    port.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_sched_q11_fake_clock_boundaries(tmp_path) -> None:
    scheduler, store, port = _build_scheduler_with_store(tmp_path)
    checkpoint = _paused_checkpoint()
    store.save(checkpoint)
    run_at = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
    store.schedule(
        ScheduledResume(
            task_id=checkpoint.task_id,
            tenant_id=checkpoint.tenant_id,
            resume_token=checkpoint.resume_token,
            run_at_utc=run_at.isoformat(),
        ),
    )
    assert await scheduler.tick(now=run_at - timedelta(seconds=1)) == 0
    port.execute.assert_not_awaited()
    assert await scheduler.tick(now=run_at) == 1
    port.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_sched_q12_cancelled_pending_not_dispatched(tmp_path) -> None:
    scheduler, store, port = _build_scheduler_with_store(tmp_path)
    checkpoint = _paused_checkpoint()
    store.save(checkpoint)
    due_at = datetime.now(timezone.utc) - timedelta(seconds=3)
    entry = store.schedule(
        ScheduledResume(
            task_id=checkpoint.task_id,
            tenant_id=checkpoint.tenant_id,
            resume_token=checkpoint.resume_token,
            run_at_utc=due_at.isoformat(),
        ),
    )
    store.cancel(entry.schedule_id)
    assert await scheduler.tick(now=datetime.now(timezone.utc)) == 0
    port.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_sched_q13_production_wiring_resumes_through_host_task_execution_port(
    tmp_path,
) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "q13_wiring.db")
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=_ok_result())
    wiring = wire_long_running_scheduler_with_host_execution(
        checkpoint_store=store,
        host_execution=port,
        poll_interval_seconds=1.0,
    )
    assert wiring is not None
    scheduler = wiring.scheduler
    checkpoint = _paused_checkpoint()
    store.save(checkpoint)
    due_at = datetime.now(timezone.utc) - timedelta(seconds=30)
    store.schedule(
        ScheduledResume(
            task_id=checkpoint.task_id,
            tenant_id=checkpoint.tenant_id,
            resume_token=checkpoint.resume_token,
            run_at_utc=due_at.isoformat(),
        ),
    )
    assert await scheduler.tick(now=datetime.now(timezone.utc)) == 1
    port.execute.assert_awaited_once()
    kwargs = port.execute.await_args.kwargs
    resumed = kwargs["resume_checkpoint"]
    assert resumed is not None
    assert resumed.task_id == checkpoint.task_id
    assert resumed.resume_token == checkpoint.resume_token
    assert resumed.runtime is not None
    assert checkpoint.runtime is not None
    assert kwargs["run_id"] == checkpoint.runtime.run_id
    assert kwargs["attempt_id"] == checkpoint.runtime.attempt_id
    assert resumed.runtime.run_id == checkpoint.runtime.run_id
    assert resumed.runtime.attempt_id == checkpoint.runtime.attempt_id


def test_sched_q14_scheduling_core_import_layer_gate() -> None:
    violations: list[str] = []
    for path in _SCHED_CORE_FILES:
        source = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_SCHED_TOKENS:
            if token in source:
                violations.append(f"{path.relative_to(_REPO_ROOT)}: forbidden token {token}")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name
                    if any(mod.startswith(p) or p.lower() in mod.lower() for p in _FORBIDDEN_SCHED_IMPORT_PREFIXES):
                        violations.append(f"{path.relative_to(_REPO_ROOT)}: import {mod}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if any(mod.startswith(p) or p.lower() in mod.lower() for p in _FORBIDDEN_SCHED_IMPORT_PREFIXES):
                    violations.append(f"{path.relative_to(_REPO_ROOT)}: from {mod}")
    assert violations == []


def test_sched_q15_no_alternate_execution_engine_in_scheduler_core() -> None:
    scheduler_source = (_SCHED_CORE_FILES[0]).read_text(encoding="utf-8")
    wiring_source = (_SCHED_CORE_FILES[-1]).read_text(encoding="utf-8")
    assert "ExecutionRuntime(" not in scheduler_source
    assert "NexusLoop(" not in scheduler_source
    assert "HostTaskResumeExecutor" in wiring_source
    assert "UnifiedTaskRunner" not in wiring_source

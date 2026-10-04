# © Artur Czarnecki. All rights reserved.

"""STATE-X-R3-R4 shared qualification helpers (SX-F13)."""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    ExecutionCheckpointEntry,
    ExecutionCheckpointStatus,
    ExecutionTreeSnapshot,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
from intergrax.runtime.long_running.scheduler import HostTaskResumeExecutor, LongRunningScheduler
from intergrax.runtime.long_running.scheduler_claim import ScheduledResumeClaim
from intergrax.runtime.long_running.scheduled_resume import (
    ScheduledResume,
    ScheduledResumePersistence,
    ScheduledResumeStatus,
)
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.utils.time_provider import SystemTimeProvider
from tests.qualification.state_x.inventory import STATE_X_FAMILY_INVENTORY

_REPO_ROOT = Path(__file__).resolve().parents[3]

_SCHEDULER_PRODUCTION_SCOPE = (
    _REPO_ROOT / "intergrax/runtime/long_running/scheduler.py",
    _REPO_ROOT / "intergrax/runtime/long_running/resume_planner.py",
    _REPO_ROOT / "intergrax/runtime/long_running/scheduled_resume.py",
    _REPO_ROOT / "intergrax/runtime/long_running/scheduled_resume_metadata.py",
    _REPO_ROOT / "intergrax/runtime/long_running/store.py",
)

_DELAYED_RESUME_FORBIDDEN_TOKENS = (
    "human_approved",
    "HumanResponseVerdict.APPROVE",
    "HumanApproverEvidence",
    "TaskHumanInput",
    "PolicyAction.ALLOW",
)


def sx_f13_inventory_entry():
    for entry in STATE_X_FAMILY_INVENTORY:
        if entry.family_id == "SX-F13":
            return entry
    raise AssertionError("SX-F13 missing from inventory")


def ok_task_result(task_id: str = "task-1") -> TaskResult:
    return TaskResult(
        task_id=task_id,
        state=TaskState.COMPLETED,
        success=True,
        authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
    )


def paused_checkpoint(
    *,
    task_id: str | None = None,
    tenant_id: str = "tenant-a",
    resume_token: str = "token-a",
    task_state: TaskState = TaskState.WAITING_FOR_RESOURCES,
) -> TaskCheckpoint:
    canonical_task_id = task_id or mint_task_id()
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
    task.state = task_state
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
        checkpoint_id=f"ckpt-{canonical_task_id[:8]}",
        task_id=canonical_task_id,
        tenant_id=tenant_id,
        resume_token=resume_token,
        task_state=task_state,
        task_snapshot=task.model_dump(mode="json"),
        progress_message="awaiting",
        created_at_utc=SystemTimeProvider.utc_now().isoformat(),
        runtime=runtime,
    )


def due_run_at(*, seconds_ago: int = 5) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()


def build_host_scheduler(
    tmp_path: Path,
    *,
    schedule_store: ScheduledResumePersistence | None = None,
) -> tuple[LongRunningScheduler, SQLiteTaskCheckpointStore, AsyncMock]:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "r3r4.db")
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=ok_task_result())
    scheduler = LongRunningScheduler(
        store,
        HostTaskResumeExecutor(port),
        schedule_store=schedule_store or store,
        ledger=store,
        owner_id="r3r4-sched",
    )
    return scheduler, store, port


class MemoryScheduleStore(ScheduledResumePersistence):
    """Replaceable ScheduledResumePersistence for R3-R4 replaceability proofs."""

    def __init__(self) -> None:
        self._rows: dict[str, ScheduledResume] = {}

    def schedule(self, entry: ScheduledResume) -> ScheduledResume:
        if entry.schedule_id in self._rows:
            from intergrax.runtime.long_running.scheduler_claim import (
                ScheduledResumeScheduleConflictError,
            )

            raise ScheduledResumeScheduleConflictError(entry.schedule_id)
        self._rows[entry.schedule_id] = entry
        return entry

    def list_due(self, *, before_utc_iso: str, limit: int = 100) -> List[ScheduledResume]:
        pending = [
            row
            for row in self._rows.values()
            if row.status is ScheduledResumeStatus.PENDING and row.run_at_utc <= before_utc_iso
        ]
        pending.sort(key=lambda r: (r.run_at_utc, r.schedule_id))
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
        for schedule_id in sorted(
            self._rows,
            key=lambda sid: (self._rows[sid].run_at_utc, sid),
        ):
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
        if row is None or row.status is not ScheduledResumeStatus.RUNNING:
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
            raise RuntimeError("missing")
        if row.status is not ScheduledResumeStatus.PENDING:
            raise RuntimeError("not pending")
        self._rows[schedule_id] = row.model_copy(update={"status": ScheduledResumeStatus.CANCELLED})


def delayed_resume_planner_source() -> str:
    return (_REPO_ROOT / "intergrax/runtime/long_running/resume_planner.py").read_text(
        encoding="utf-8",
    )


def build_scheduled_resume_task_has_no_metadata_hitl() -> bool:
    tree = ast.parse(delayed_resume_planner_source())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "build_scheduled_resume_task":
            continue
        segment = ast.get_source_segment(delayed_resume_planner_source(), node) or ""
        return not any(token in segment for token in _DELAYED_RESUME_FORBIDDEN_TOKENS)
    return False


def scheduler_scope_authority_negative_hits() -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    planner_path = _REPO_ROOT / "intergrax/runtime/long_running/resume_planner.py"
    text = planner_path.read_text(encoding="utf-8")
    fn_start = text.find("def build_scheduled_resume_task")
    fn_end = text.find("\ndef build_checkpoint_resume_task")
    delayed = text[fn_start:fn_end] if fn_start >= 0 else ""
    rel = planner_path.relative_to(_REPO_ROOT).as_posix()
    for token in _DELAYED_RESUME_FORBIDDEN_TOKENS:
        if token in delayed:
            hits.append((rel, token))
    scheduler_text = (
        _REPO_ROOT / "intergrax/runtime/long_running/scheduler.py"
    ).read_text(encoding="utf-8")
    for token in ("HumanApproverEvidence", "TaskHumanInput"):
        if token in scheduler_text:
            hits.append(("intergrax/runtime/long_running/scheduler.py", token))
    return hits

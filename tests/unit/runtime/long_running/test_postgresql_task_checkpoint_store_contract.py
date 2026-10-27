# © Artur Czarnecki. All rights reserved.

"""PostgreSQL task checkpoint store — HEP contract negatives (P4-A C04)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tests.qualification.dist_scale_x._p2_support import pg_port_open

pytestmark = [
    pytest.mark.skipif(not pg_port_open(), reason="PostgreSQL qual infra unavailable"),
    pytest.mark.unit,
]


def _pg_store(suffix: str):
    import uuid

    schema = f"dist_scale_x_p4_{suffix}_{uuid.uuid4().hex[:8]}"
    from intergrax.integrations.providers.relational_store.postgresql.config import (
        PostgreSQLIntegrationConfig,
    )
    from intergrax.runtime.long_running.postgresql_task_checkpoint_store import (
        PostgreSQLTaskCheckpointStore,
    )

    config = PostgreSQLIntegrationConfig.from_env().model_copy(update={"tenant_schema": schema})
    return PostgreSQLTaskCheckpointStore(config=config)


def _task(tenant: str = "t-pg", task_id: str | None = None):
    from intergrax.runtime.task.task import Task, TaskState
    from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions

    task = Task(
        tenant_id=tenant,
        user_id="u1",
        message="pg",
        state=TaskState.WAITING_FOR_HUMAN,
        options=TaskExecutionOptions(long_running=TaskLongRunningOptions(enabled=True)),
    )
    if task_id:
        task.task_id = task_id
    return task


def _persist(task, store, run_id, attempt_id):
    from intergrax.runtime.long_running.coordinator import LongRunningCoordinator

    return LongRunningCoordinator.persist_checkpoint(task, store, run_id=run_id, attempt_id=attempt_id)


def _identity():
    from intergrax.contracts.execution_identity import (
        bind_active_execution_identity,
        mint_attempt_id,
        mint_execution_id,
        mint_run_id,
        reset_active_execution_identity,
    )

    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
    )
    return run_id, attempt_id, token


def test_pg_idempotent_checkpoint_replay() -> None:
    from intergrax.contracts.execution_identity import reset_active_execution_identity
    from intergrax.runtime.long_running.checkpoint_builder import build_task_checkpoint
    from intergrax.runtime.long_running.checkpoint_builder import build_runtime_checkpoint

    store = _pg_store("idempotent")
    task = _task()
    run_id, attempt_id, token = _identity()
    try:
        first = _persist(task, store, run_id, attempt_id)
        replayed = store.save(first, expected_revision=first.revision)
        assert replayed.checkpoint_id == first.checkpoint_id
    finally:
        reset_active_execution_identity(token)


def test_pg_checkpoint_id_conflict() -> None:
    from intergrax.contracts.execution_identity import reset_active_execution_identity
    from intergrax.runtime.long_running.checkpoint_revision import CheckpointIdConflictError
    from intergrax.runtime.long_running.checkpoint_builder import build_task_checkpoint, build_runtime_checkpoint

    store = _pg_store("id_conflict")
    task = _task()
    run_id, attempt_id, token = _identity()
    try:
        first = _persist(task, store, run_id, attempt_id)
        runtime = build_runtime_checkpoint(task, run_id=run_id, attempt_id=attempt_id)
        conflicting = build_task_checkpoint(
            task,
            runtime=runtime,
            resume_token=first.resume_token,
            progress_message="different-payload",
        )
        conflicting.checkpoint_id = first.checkpoint_id
        with pytest.raises(CheckpointIdConflictError):
            store.save(conflicting, expected_revision=first.revision)
    finally:
        reset_active_execution_identity(token)


def test_pg_tenant_isolation() -> None:
    from intergrax.contracts.execution_identity import reset_active_execution_identity

    store = _pg_store("tenant_iso")
    task_a = _task("tenant-a")
    task_b = _task("tenant-b")
    run_id, attempt_id, token = _identity()
    try:
        saved = _persist(task_a, store, run_id, attempt_id)
        assert store.get_latest(saved.task_id, "tenant-b") is None
    finally:
        reset_active_execution_identity(token)


def test_pg_scheduled_resume_claim_and_complete() -> None:
    from intergrax.runtime.long_running.scheduled_resume import ScheduledResume

    store = _pg_store("schedule")
    run_at = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    entry = store.schedule(
        ScheduledResume(
            task_id="task-sched",
            tenant_id="tenant-s",
            resume_token="tok-sched",
            run_at_utc=run_at,
        ),
    )
    claims = store.claim_due(
        before_utc_iso=datetime.now(timezone.utc).isoformat(),
        owner_id="owner-a",
        lease_seconds=300,
    )
    assert len(claims) == 1
    store.complete_claim(claims[0])
    due = store.list_due(before_utc_iso=datetime.now(timezone.utc).isoformat())
    assert due == []


def test_pg_scheduler_ledger_single_owner() -> None:
    import uuid

    store = _pg_store("ledger")
    key = f"ledger-key-{uuid.uuid4().hex}"
    claim_a = store.claim_action(key, "owner-a", 300, action="tick")
    assert claim_a is not None
    claim_b = store.claim_action(key, "owner-b", 300, action="tick")
    assert claim_b is None
    store.complete_action(claim_a)
    assert store.has_action(key)


def test_pg_terminal_put_if_absent() -> None:
    from intergrax.contracts.execution_terminal import ExecutionTerminalOutcome, ExecutionTerminalRecord

    import uuid

    store = _pg_store("terminal")
    task_id = f"task-terminal-{uuid.uuid4().hex[:8]}"
    record = ExecutionTerminalRecord(
        tenant_id="tenant-t",
        task_id=task_id,
        outcome=ExecutionTerminalOutcome.COMPLETED,
        reason="done",
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    assert store.put_terminal_record_if_absent(record) is True
    assert store.put_terminal_record_if_absent(record) is False
    loaded = store.get_terminal_record(tenant_id="tenant-t", task_id=task_id)
    assert loaded is not None


def test_pg_concurrent_schema_init() -> None:
    import threading

    errors: list[Exception] = []

    def _init() -> None:
        try:
            _pg_store("schema_race")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_init) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors

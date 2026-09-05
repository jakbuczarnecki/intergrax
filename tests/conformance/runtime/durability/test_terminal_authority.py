# © Artur Czarnecki. All rights reserved.

"""P0C-5 + P0C-6 — terminal authority durability across process restart."""

from __future__ import annotations

import json
import threading

import pytest

from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id, mint_task_id
from intergrax.contracts.execution_terminal import (
    ExecutionTerminalConflictError,
    ExecutionTerminalError,
    ExecutionTerminalOutcome,
    ExecutionTerminalRecord,
)
from intergrax.runtime.cancellation.resume_admission import (
    CheckpointNotResumableError,
    assert_checkpoint_resumable,
    is_checkpoint_resumable,
)
from intergrax.runtime.execution.execution_terminal import ExecutionTerminalService
from intergrax.runtime.execution.execution_terminal.persistence import (
    DocumentStoreExecutionTerminalStore,
    KvExecutionTerminalStore,
    decode_terminal_record,
    encode_terminal_record,
    normalize_terminal_record,
)
from intergrax.runtime.long_running.execution_tree_checkpoint import minimal_runtime_checkpoint
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions

from tests.conformance.runtime.durability.provider_factories import (
    DurableAdmissionBacking,
    create_admission_dependencies,
)
from tests.conformance.runtime.durability.restart import (
    fresh_admission_composition,
    fresh_checkpoint_composition,
)
from tests.unit.runtime.background_execution.reentry_admission_doubles import InMemoryKVStore
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-p0c8-terminal"


def _paused_checkpoint() -> TaskCheckpoint:
    task_id = str(mint_task_id())
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    return TaskCheckpoint(
        task_id=task_id,
        tenant_id=_TENANT,
        resume_token="rt-terminal",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=Task(
            task_id=task_id,
            tenant_id=_TENANT,
            user_id="user",
            message="paused",
            state=TaskState.WAITING_FOR_HUMAN,
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(enabled=True, resume_token="rt-terminal"),
            ),
        ).model_dump(mode="json"),
        runtime=minimal_runtime_checkpoint(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            root_execution_id=mint_execution_id(),
        ),
    )


@pytest.mark.parametrize(
    ("store_a", "store_b"),
    [
        (
            lambda: KvExecutionTerminalStore(InMemoryKVStore()),
            lambda store: KvExecutionTerminalStore(store._kv_store),  # noqa: SLF001
        ),
        (
            lambda: DocumentStoreExecutionTerminalStore(InMemoryDocumentStore()),
            lambda store: DocumentStoreExecutionTerminalStore(store._document_store),  # noqa: SLF001
        ),
    ],
    ids=["kv", "document_store"],
)
def test_terminal_codec_round_trip_survives_restart(store_a, store_b) -> None:
    adapter_a = store_a()
    record = normalize_terminal_record(
        ExecutionTerminalRecord(
            tenant_id=_TENANT,
            task_id=str(mint_task_id()),
            run_id=None,
            outcome=ExecutionTerminalOutcome.COMPLETED,
            reason="completed",
            recorded_at_utc="2026-01-01T00:00:00Z",
        )
    )
    adapter_a.put_if_absent(record)
    adapter_b = store_b(adapter_a)
    loaded = adapter_b.load_record(tenant_id=record.tenant_id, task_id=record.task_id)
    assert loaded == record
    assert decode_terminal_record(encode_terminal_record(record)) == record


def test_terminal_codec_unsupported_schema_version_fails_closed() -> None:
    record = normalize_terminal_record(
        ExecutionTerminalRecord(
            tenant_id=_TENANT,
            task_id=str(mint_task_id()),
            run_id=None,
            outcome=ExecutionTerminalOutcome.FAILED,
            reason="failed",
            recorded_at_utc="2026-01-01T00:00:00Z",
        )
    )
    payload = json.loads(encode_terminal_record(record).decode("utf-8"))
    payload["schema_version"] = 999
    corrupted = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    with pytest.raises(ExecutionTerminalError, match="unsupported execution terminal schema"):
        decode_terminal_record(corrupted)


@pytest.mark.parametrize(
    "winner",
    [
        ExecutionTerminalOutcome.COMPLETED,
        ExecutionTerminalOutcome.FAILED,
        ExecutionTerminalOutcome.CANCELLED,
    ],
)
def test_terminal_winner_survives_restart(
    admission_backing: DurableAdmissionBacking,
    winner: ExecutionTerminalOutcome,
) -> None:
    process_a = create_admission_dependencies(admission_backing)
    task_id = str(mint_task_id())
    run_id = mint_run_id()
    process_a.execution_terminal.commit_terminal_outcome(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        outcome=winner,
        reason=winner.value,
    )
    process_b = fresh_admission_composition(admission_backing)
    loaded = process_b.execution_terminal.get_terminal_record(tenant_id=_TENANT, task_id=task_id)
    assert loaded is not None
    assert loaded.outcome is winner

    losers = {
        ExecutionTerminalOutcome.COMPLETED,
        ExecutionTerminalOutcome.FAILED,
        ExecutionTerminalOutcome.CANCELLED,
    } - {winner}
    for loser in losers:
        with pytest.raises(ExecutionTerminalConflictError):
            process_b.execution_terminal.commit_terminal_outcome(
                tenant_id=_TENANT,
                task_id=task_id,
                run_id=run_id,
                outcome=loser,
                reason=loser.value,
            )


def test_failed_terminal_idempotent_after_restart(
    admission_backing: DurableAdmissionBacking,
) -> None:
    process_a = create_admission_dependencies(admission_backing)
    task_id = str(mint_task_id())
    first = process_a.execution_terminal.commit_terminal_outcome(
        tenant_id=_TENANT,
        task_id=task_id,
        outcome=ExecutionTerminalOutcome.FAILED,
        reason="graph_failed",
    )
    process_b = fresh_admission_composition(admission_backing)
    second = process_b.execution_terminal.commit_terminal_outcome(
        tenant_id=_TENANT,
        task_id=task_id,
        outcome=ExecutionTerminalOutcome.FAILED,
        reason="late_duplicate",
    )
    assert second.outcome == first.outcome
    assert second.recorded_at_utc == first.recorded_at_utc


@pytest.mark.parametrize(
    "outcome",
    [
        ExecutionTerminalOutcome.COMPLETED,
        ExecutionTerminalOutcome.FAILED,
        ExecutionTerminalOutcome.CANCELLED,
    ],
)
def test_terminal_outcome_blocks_resume_after_restart(tmp_path, outcome: ExecutionTerminalOutcome) -> None:
    db_path = tmp_path / f"terminal-resume-{outcome.value}.db"
    store_a, terminal_a = fresh_checkpoint_composition(db_path)
    checkpoint = _paused_checkpoint()
    store_a.save(checkpoint)
    terminal_a.commit_terminal_outcome(
        tenant_id=checkpoint.tenant_id,
        task_id=checkpoint.task_id,
        run_id=checkpoint.runtime.run_id if checkpoint.runtime else None,
        outcome=outcome,
        reason=outcome.value,
    )

    _, terminal_b = fresh_checkpoint_composition(db_path)
    store_b = store_a.__class__(db_path=db_path)
    loaded = store_b.get_by_token(checkpoint.task_id, _TENANT, checkpoint.resume_token)
    assert loaded is not None
    assert is_checkpoint_resumable(loaded, execution_terminal=terminal_b) is False
    with pytest.raises(CheckpointNotResumableError):
        assert_checkpoint_resumable(loaded, execution_terminal=terminal_b)


def test_cancellation_blocks_stale_checkpoint_after_restart(tmp_path) -> None:
    db_path = tmp_path / "cancel-resume.db"
    store_a, terminal_a = fresh_checkpoint_composition(db_path)
    checkpoint = _paused_checkpoint()
    store_a.save(checkpoint)
    terminal_a.record_cancellation(
        tenant_id=checkpoint.tenant_id,
        task_id=checkpoint.task_id,
        run_id=checkpoint.runtime.run_id if checkpoint.runtime else None,
        reason="operator_cancel",
    )

    _, terminal_b = fresh_checkpoint_composition(db_path)
    store_b = store_a.__class__(db_path=db_path)
    loaded = store_b.get_by_token(checkpoint.task_id, _TENANT, checkpoint.resume_token)
    assert loaded is not None
    assert is_checkpoint_resumable(loaded, execution_terminal=terminal_b) is False


def test_terminal_race_winner_survives_restart() -> None:
    backing = DurableAdmissionBacking.fresh_kv()
    process_a = create_admission_dependencies(backing)
    task_id = str(mint_task_id())
    barrier = threading.Barrier(2)
    winners: list[ExecutionTerminalRecord] = []
    conflicts: list[ExecutionTerminalConflictError] = []

    def worker(outcome: ExecutionTerminalOutcome) -> None:
        barrier.wait()
        try:
            winners.append(
                process_a.execution_terminal.commit_terminal_outcome(
                    tenant_id=_TENANT,
                    task_id=task_id,
                    outcome=outcome,
                    reason=outcome.value,
                )
            )
        except ExecutionTerminalConflictError as exc:
            conflicts.append(exc)

    t1 = threading.Thread(
        target=worker,
        args=(ExecutionTerminalOutcome.COMPLETED,),
    )
    t2 = threading.Thread(
        target=worker,
        args=(ExecutionTerminalOutcome.FAILED,),
    )
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(winners) == 1

    process_b = fresh_admission_composition(backing)
    canonical = process_b.execution_terminal.get_terminal_record(tenant_id=_TENANT, task_id=task_id)
    assert canonical is not None
    assert canonical.outcome == winners[0].outcome

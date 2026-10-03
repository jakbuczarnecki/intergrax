# © Artur Czarnecki. All rights reserved.

"""STATE-X-R1 — checkpoint resume, terminal composition, snapshot authority closure."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_terminal import (
    ExecutionTerminalError,
    ExecutionTerminalOutcome,
    ExecutionTerminalPersistenceCapability,
    ExecutionTerminalRecord,
)
from intergrax.contracts.structured_json_value import JsonObject
from intergrax.runtime.execution.execution_terminal import (
    CheckpointStoreExecutionTerminalStore,
    ExecutionTerminalService,
    InMemoryExecutionTerminalStore,
    wire_execution_terminal_store,
)
from intergrax.runtime.execution.execution_terminal.durability_policy import (
    DURABLE_EXECUTION_TERMINAL_REQUIRED_MSG,
    validate_durable_execution_terminal_for_composition,
)
from intergrax.runtime.long_running.checkpoint_resume_validation import (
    CheckpointResumeEligibility,
    CheckpointResumeValidationError,
    evaluate_checkpoint_resume_eligibility,
    resolve_resume_execution_authority,
    validate_checkpoint_authority_expansion,
    validate_checkpoint_resume_authority,
    validate_checkpoint_snapshot_integrity,
)
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.execution_tree_checkpoint import minimal_runtime_checkpoint
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TENANT = "tenant-state-x-r1"
_NEXUS_SOURCE = _REPO_ROOT / "intergrax/runtime/nexus/nexus_loop.py"
_VALIDATION_SOURCE = (
    _REPO_ROOT / "intergrax/runtime/long_running/checkpoint_resume_validation.py"
)


def _paused_checkpoint(
    *,
    task_id: str | None = None,
    tenant_id: str = _TENANT,
    execution_authority: ParentExecutionAuthority | None = None,
    revision: int | None = 1,
) -> TaskCheckpoint:
    resolved_task_id = task_id or str(mint_task_id())
    task = Task(
        task_id=resolved_task_id,
        tenant_id=tenant_id,
        user_id="user",
        message="paused",
        state=TaskState.WAITING_FOR_HUMAN,
        execution_authority=execution_authority,
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, resume_token="rt-r1"),
        ),
    )
    return TaskCheckpoint(
        task_id=resolved_task_id,
        tenant_id=tenant_id,
        resume_token="rt-r1",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=task.model_dump(mode="json"),
        revision=revision,
        runtime=minimal_runtime_checkpoint(
            task_id=resolved_task_id,
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            root_execution_id=mint_execution_id(),
        ),
    )


def _evaluate(
    checkpoint: TaskCheckpoint,
    *,
    task_id: str | None = None,
    tenant_id: str = _TENANT,
    target_run_id: str | None = None,
    target_attempt_id: str | None = None,
    target_root_execution_id: str | None = None,
    latest_checkpoint: TaskCheckpoint | None = None,
    execution_terminal: ExecutionTerminalService | None = None,
    current_task: Task | None = None,
) -> CheckpointResumeEligibility:
    return evaluate_checkpoint_resume_eligibility(
        checkpoint,
        target_task_id=task_id or checkpoint.task_id,
        target_tenant_id=tenant_id,
        target_run_id=target_run_id,
        target_attempt_id=target_attempt_id,
        target_root_execution_id=target_root_execution_id,
        latest_checkpoint=latest_checkpoint,
        execution_terminal=execution_terminal,
        current_task=current_task,
    ).eligibility


def _resume_task(
    checkpoint: TaskCheckpoint,
    *,
    execution_authority: ParentExecutionAuthority | None = None,
) -> Task:
    return Task(
        task_id=checkpoint.task_id,
        tenant_id=checkpoint.tenant_id,
        user_id="user",
        message="resume",
        execution_authority=execution_authority,
        state=TaskState.CREATED,
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(
                enabled=True,
                resume_token=checkpoint.resume_token,
            ),
        ),
    )


def test_r1_q01_task_snapshot_uses_json_object_boundary() -> None:
    source = (_REPO_ROOT / "intergrax/runtime/long_running/models.py").read_text(
        encoding="utf-8",
    )
    assert "task_snapshot: Dict[str, Any]" not in source
    assert "task_snapshot: JsonObject" in source
    assert "from intergrax.contracts.structured_json_value import JsonObject" in source
    sample = TaskCheckpoint(
        task_id=str(mint_task_id()),
        tenant_id=_TENANT,
        resume_token="rt",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot={"task_id": "x", "tenant_id": _TENANT, "user_id": "u"},
    )
    assert isinstance(sample.task_snapshot, dict)


def test_r1_q02_missing_snapshot_task_id_reject_malformed() -> None:
    checkpoint = _paused_checkpoint()
    snap = dict(checkpoint.task_snapshot)
    del snap["task_id"]
    result = _evaluate(checkpoint.model_copy(update={"task_snapshot": snap}))
    assert result is CheckpointResumeEligibility.REJECT_MALFORMED


def test_r1_q03_missing_snapshot_tenant_id_reject_malformed() -> None:
    checkpoint = _paused_checkpoint()
    snap = dict(checkpoint.task_snapshot)
    del snap["tenant_id"]
    result = _evaluate(checkpoint.model_copy(update={"task_snapshot": snap}))
    assert result is CheckpointResumeEligibility.REJECT_MALFORMED


def test_r1_q04_empty_snapshot_reject_malformed() -> None:
    checkpoint = _paused_checkpoint().model_copy(update={"task_snapshot": {}})
    assert _evaluate(checkpoint) is CheckpointResumeEligibility.REJECT_MALFORMED


def test_r1_q05_malformed_snapshot_reject_malformed() -> None:
    checkpoint = _paused_checkpoint().model_copy(
        update={"task_snapshot": {"task_id": "x", "tenant_id": _TENANT, "state": 99999}},
    )
    assert _evaluate(checkpoint) is CheckpointResumeEligibility.REJECT_MALFORMED


def test_r1_q06_snapshot_task_mismatch_reject_identity() -> None:
    checkpoint = _paused_checkpoint()
    other = str(mint_task_id())
    snap = dict(checkpoint.task_snapshot)
    snap["task_id"] = other
    result = _evaluate(checkpoint.model_copy(update={"task_snapshot": snap}))
    assert result is CheckpointResumeEligibility.REJECT_IDENTITY


def test_r1_q07_snapshot_tenant_mismatch_reject_tenant() -> None:
    checkpoint = _paused_checkpoint()
    snap = dict(checkpoint.task_snapshot)
    snap["tenant_id"] = "tenant-other"
    result = _evaluate(checkpoint.model_copy(update={"task_snapshot": snap}))
    assert result is CheckpointResumeEligibility.REJECT_TENANT


def test_r1_q08_outer_checkpoint_tenant_mismatch_reject_tenant() -> None:
    checkpoint = _paused_checkpoint()
    assert _evaluate(checkpoint, tenant_id="wrong-tenant") is CheckpointResumeEligibility.REJECT_TENANT


def test_r1_q09_outer_task_mismatch_reject_identity() -> None:
    checkpoint = _paused_checkpoint()
    assert _evaluate(checkpoint, task_id=str(mint_task_id())) is CheckpointResumeEligibility.REJECT_IDENTITY


def test_r1_q10_runtime_identity_mismatch_reject_identity() -> None:
    checkpoint = _paused_checkpoint()
    assert _evaluate(checkpoint, target_run_id=mint_run_id()) is CheckpointResumeEligibility.REJECT_IDENTITY


def test_r1_q11_stale_revision_reject_stale() -> None:
    older = _paused_checkpoint(revision=1)
    newer = _paused_checkpoint(task_id=older.task_id, revision=2)
    assert _evaluate(older, latest_checkpoint=newer) is CheckpointResumeEligibility.REJECT_STALE


def test_r1_q12_terminal_record_blocks_resume(tmp_path: Path) -> None:
    checkpoint = _paused_checkpoint()
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "term.db")
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
    loaded = store.get_latest(checkpoint.task_id, _TENANT)
    assert loaded is not None
    assert _evaluate(loaded, execution_terminal=terminal) is CheckpointResumeEligibility.REJECT_TERMINAL


def test_r1_q13_snapshot_cannot_override_terminal(tmp_path: Path) -> None:
    checkpoint = _paused_checkpoint()
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "dom.db")
    store.save(checkpoint)
    terminal = ExecutionTerminalService(CheckpointStoreExecutionTerminalStore(store))
    assert checkpoint.runtime is not None
    terminal.commit_terminal_outcome(
        tenant_id=checkpoint.tenant_id,
        task_id=checkpoint.task_id,
        run_id=checkpoint.runtime.run_id,
        outcome=ExecutionTerminalOutcome.CANCELLED,
        reason="cancelled",
    )
    loaded = store.get_latest(checkpoint.task_id, _TENANT)
    assert loaded is not None
    assert loaded.task_state is TaskState.WAITING_FOR_HUMAN
    assert _evaluate(loaded, execution_terminal=terminal) in (
        CheckpointResumeEligibility.REJECT_TERMINAL,
        CheckpointResumeEligibility.REJECT_CANCELLED,
    )


def test_r1_q14_historical_authority_cannot_exceed_current() -> None:
    checkpoint = _paused_checkpoint(
        execution_authority=ParentExecutionAuthority.scoped(("read", "write")),
    )
    current = _resume_task(
        checkpoint,
        execution_authority=ParentExecutionAuthority.scoped(("read",)),
    )
    effective = resolve_resume_execution_authority(checkpoint, current)
    assert effective == ParentExecutionAuthority.scoped(("read",))


def test_r1_q15_current_authority_absent_reject_authority() -> None:
    checkpoint = _paused_checkpoint(
        execution_authority=ParentExecutionAuthority.scoped(("read",)),
    )
    current = _resume_task(checkpoint, execution_authority=None)
    assert (
        validate_checkpoint_resume_authority(checkpoint, current).eligibility
        is CheckpointResumeEligibility.REJECT_AUTHORITY
    )


def test_r1_q16_valid_snapshot_unknown_authority_not_malformed() -> None:
    checkpoint = _paused_checkpoint(execution_authority=None)
    current = _resume_task(
        checkpoint,
        execution_authority=ParentExecutionAuthority.scoped(("read",)),
    )
    snap = validate_checkpoint_snapshot_integrity(checkpoint)
    assert snap.eligibility is CheckpointResumeEligibility.ALLOW_RESUME
    effective = resolve_resume_execution_authority(checkpoint, current)
    assert effective == ParentExecutionAuthority.scoped(("read",))


def test_r1_q17_rejected_snapshot_zero_task_mutation(tmp_path: Path) -> None:
    checkpoint = _paused_checkpoint().model_copy(update={"task_snapshot": {}})
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "nomut.db")
    store.save(checkpoint)
    task = _resume_task(checkpoint, execution_authority=ParentExecutionAuthority.scoped(("read",)))
    before = task.model_dump()
    with pytest.raises(CheckpointResumeValidationError):
        LongRunningCoordinator.restore_if_resuming(task, store)
    assert task.model_dump() == before


class _FakeTerminalCapableCheckpointStore(TaskCheckpointPersistence):
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], ExecutionTerminalRecord] = {}

    def list_for_task(self, task_id: str, tenant_id: str) -> list[TaskCheckpoint]:
        return []

    def get_latest(self, task_id: str, tenant_id: str) -> TaskCheckpoint | None:
        return None

    def get_by_token(
        self,
        task_id: str,
        tenant_id: str,
        resume_token: str,
    ) -> TaskCheckpoint | None:
        return None

    def list_paused(self) -> list[TaskCheckpoint]:
        return []

    def save(
        self,
        checkpoint: TaskCheckpoint,
        *,
        expected_revision: int | None = None,
    ) -> TaskCheckpoint:
        return checkpoint

    def get_terminal_record(self, *, tenant_id: str, task_id: str) -> ExecutionTerminalRecord | None:
        return self._records.get((tenant_id, task_id))

    def put_terminal_record_if_absent(self, record: ExecutionTerminalRecord) -> bool:
        key = (record.tenant_id, record.task_id)
        if key in self._records:
            return False
        self._records[key] = record
        return True


class _FakeCheckpointStoreWithoutTerminalCapability(TaskCheckpointPersistence):
    def list_for_task(self, task_id: str, tenant_id: str) -> list[TaskCheckpoint]:
        return []

    def get_latest(self, task_id: str, tenant_id: str) -> TaskCheckpoint | None:
        return None

    def get_by_token(
        self,
        task_id: str,
        tenant_id: str,
        resume_token: str,
    ) -> TaskCheckpoint | None:
        return None

    def list_paused(self) -> list[TaskCheckpoint]:
        return []

    def save(
        self,
        checkpoint: TaskCheckpoint,
        *,
        expected_revision: int | None = None,
    ) -> TaskCheckpoint:
        return checkpoint


def test_r1_q18_custom_terminal_capable_provider() -> None:
    store = _FakeTerminalCapableCheckpointStore()
    assert isinstance(store, ExecutionTerminalPersistenceCapability)
    wired = wire_execution_terminal_store(checkpoint_store=store)
    assert isinstance(wired, CheckpointStoreExecutionTerminalStore)
    assert wired.is_durable is True


def test_r1_q19_non_terminal_checkpoint_provider_not_widened() -> None:
    store = _FakeCheckpointStoreWithoutTerminalCapability()
    assert not isinstance(store, ExecutionTerminalPersistenceCapability)
    wired = wire_execution_terminal_store(checkpoint_store=store)
    assert isinstance(wired, InMemoryExecutionTerminalStore)
    assert wired.is_durable is False


def test_r1_q20_production_without_durable_terminal_fails_closed() -> None:
    store = _FakeCheckpointStoreWithoutTerminalCapability()
    terminal_store = wire_execution_terminal_store(checkpoint_store=store)
    with pytest.raises(ExecutionTerminalError, match=DURABLE_EXECUTION_TERMINAL_REQUIRED_MSG):
        validate_durable_execution_terminal_for_composition(
            production_mode=True,
            checkpoint_store=store,
            store=terminal_store,
        )


def test_r1_q21_nexus_composition_no_cast_or_probing() -> None:
    source = _NEXUS_SOURCE.read_text(encoding="utf-8")
    assert "cast(" not in source
    assert "type: ignore" not in source
    assert "SQLiteTaskCheckpointStore" not in source
    assert not re.search(r"\b(getattr|hasattr)\(", source)
    assert "ExecutionTerminalPersistenceCapability" in source
    assert "terminal_checkpoint_capability" in source


def test_r1_q22_single_snapshot_parser_semantics() -> None:
    source = _VALIDATION_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    validate_calls = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "model_validate":
            if isinstance(func.value, ast.Name) and func.value.id == "Task":
                if node.args and isinstance(node.args[0], ast.Attribute):
                    attr = node.args[0]
                    if (
                        isinstance(attr.value, ast.Name)
                        and attr.value.id == "checkpoint"
                        and attr.attr == "task_snapshot"
                    ):
                        validate_calls += 1
    assert validate_calls == 0
    assert "_parse_checkpoint_snapshot_task" in source


def test_r1_blank_tenant_in_snapshot_reject_malformed() -> None:
    checkpoint = _paused_checkpoint()
    snap = dict(checkpoint.task_snapshot)
    snap["tenant_id"] = "   "
    assert (
        _evaluate(checkpoint.model_copy(update={"task_snapshot": snap}))
        is CheckpointResumeEligibility.REJECT_MALFORMED
    )


def test_r1_authority_expansion_uses_canonical_parser() -> None:
    checkpoint = _paused_checkpoint().model_copy(update={"task_snapshot": {}})
    result = validate_checkpoint_authority_expansion(
        checkpoint,
        ParentExecutionAuthority.scoped(("read",)),
    )
    assert result.eligibility is CheckpointResumeEligibility.REJECT_MALFORMED

# © Artur Czarnecki. All rights reserved.

"""P0C-8A — single terminal authority across background + checkpoint/resume consumers."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from intergrax.contracts.attempt_lifecycle import AttemptTransitionReason
from intergrax.contracts.execution_identity import mint_execution_id
from intergrax.contracts.execution_terminal import (
    ExecutionTerminalConflictError,
    ExecutionTerminalOutcome,
    ExecutionTerminalRecord,
    ExecutionTerminalStore,
)
from intergrax.runtime.background_execution.reentry_admission import (
    BackgroundExecutionReentryDisposition,
)
from intergrax.runtime.cancellation.resume_admission import (
    CheckpointNotResumableError,
    assert_checkpoint_resumable,
)
from intergrax.runtime.long_running.execution_tree_checkpoint import minimal_runtime_checkpoint
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions

from tests.conformance.runtime.durability._helpers import admit, transport_ref
from tests.conformance.runtime.durability.provider_factories import (
    DurableAdmissionBacking,
    create_admission_dependencies,
    create_checkpoint_store,
    create_shared_terminal_composition,
)
from tests.conformance.runtime.durability.restart import (
    fresh_admission_composition,
    fresh_shared_checkpoint_composition,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _paused_checkpoint(
    *,
    tenant_id: str,
    task_id: str,
    run_id,
    attempt_id,
    resume_token: str = "rt-integrated",
) -> TaskCheckpoint:
    return TaskCheckpoint(
        task_id=task_id,
        tenant_id=tenant_id,
        resume_token=resume_token,
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=Task(
            task_id=task_id,
            tenant_id=tenant_id,
            user_id="user",
            message="paused",
            state=TaskState.WAITING_FOR_HUMAN,
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(enabled=True, resume_token=resume_token),
            ),
        ).model_dump(mode="json"),
        runtime=minimal_runtime_checkpoint(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            root_execution_id=mint_execution_id(),
        ),
    )


def test_background_terminal_blocks_checkpoint_resume_without_second_commit(
    admission_backing: DurableAdmissionBacking,
    tenant_id: str,
    tmp_path,
) -> None:
    db_path = tmp_path / "shared-terminal-cancel.db"
    process_a, checkpoint_store = create_shared_terminal_composition(admission_backing, db_path)
    transport = transport_ref(tenant_id=tenant_id, task_id="shared-cancel")
    first = admit(transport=transport, deps=process_a)
    checkpoint = _paused_checkpoint(
        tenant_id=tenant_id,
        task_id=str(first.identity.task_id),
        run_id=first.identity.run_id,
        attempt_id=first.identity.attempt_id,
    )
    checkpoint_store.save(checkpoint)

    process_a.execution_terminal.commit_terminal_outcome(
        tenant_id=first.identity.tenant_id,
        task_id=str(first.identity.task_id),
        run_id=first.identity.run_id,
        outcome=ExecutionTerminalOutcome.CANCELLED,
        reason="operator_cancel",
    )

    _, resume_terminal = fresh_shared_checkpoint_composition(admission_backing, db_path)
    loaded = create_checkpoint_store(db_path).get_by_token(
        checkpoint.task_id,
        tenant_id,
        checkpoint.resume_token,
    )
    assert loaded is not None
    with pytest.raises(CheckpointNotResumableError):
        assert_checkpoint_resumable(loaded, execution_terminal=resume_terminal)


def test_checkpoint_or_nexus_terminal_blocks_background_redelivery_without_second_commit(
    admission_backing: DurableAdmissionBacking,
    tenant_id: str,
    tmp_path,
) -> None:
    db_path = tmp_path / "shared-terminal-completed.db"
    process_a, checkpoint_store = create_shared_terminal_composition(admission_backing, db_path)
    transport = transport_ref(tenant_id=tenant_id, task_id="shared-completed")
    first = admit(transport=transport, deps=process_a)
    checkpoint = _paused_checkpoint(
        tenant_id=tenant_id,
        task_id=str(first.identity.task_id),
        run_id=first.identity.run_id,
        attempt_id=first.identity.attempt_id,
    )
    checkpoint_store.save(checkpoint)

    process_a.execution_terminal.commit_terminal_outcome(
        tenant_id=first.identity.tenant_id,
        task_id=str(first.identity.task_id),
        run_id=first.identity.run_id,
        outcome=ExecutionTerminalOutcome.COMPLETED,
        reason="nexus_finalized",
    )

    process_b = fresh_admission_composition(admission_backing)
    redelivery = admit(transport=transport, deps=process_b)
    assert redelivery.disposition is BackgroundExecutionReentryDisposition.TERMINAL_ALREADY_RECORDED


def test_shared_terminal_authority_survives_restart(
    admission_backing: DurableAdmissionBacking,
    tenant_id: str,
    tmp_path,
) -> None:
    db_path = tmp_path / "shared-terminal-restart.db"
    process_a, checkpoint_store = create_shared_terminal_composition(admission_backing, db_path)
    transport = transport_ref(tenant_id=tenant_id, task_id="shared-restart")
    first = admit(transport=transport, deps=process_a)
    checkpoint = _paused_checkpoint(
        tenant_id=tenant_id,
        task_id=str(first.identity.task_id),
        run_id=first.identity.run_id,
        attempt_id=first.identity.attempt_id,
    )
    checkpoint_store.save(checkpoint)
    process_a.execution_terminal.commit_terminal_outcome(
        tenant_id=first.identity.tenant_id,
        task_id=str(first.identity.task_id),
        run_id=first.identity.run_id,
        outcome=ExecutionTerminalOutcome.CANCELLED,
        reason="operator_cancel",
    )

    process_b = fresh_admission_composition(admission_backing)
    _, resume_terminal = fresh_shared_checkpoint_composition(admission_backing, db_path)
    redelivery = admit(transport=transport, deps=process_b)
    loaded = create_checkpoint_store(db_path).get_by_token(
        checkpoint.task_id,
        tenant_id,
        checkpoint.resume_token,
    )
    assert loaded is not None
    assert redelivery.disposition is BackgroundExecutionReentryDisposition.TERMINAL_ALREADY_RECORDED
    with pytest.raises(CheckpointNotResumableError):
        assert_checkpoint_resumable(loaded, execution_terminal=resume_terminal)


def test_shared_terminal_authority_rejects_cross_consumer_conflict(
    admission_backing: DurableAdmissionBacking,
    tenant_id: str,
) -> None:
    process_a = create_admission_dependencies(admission_backing)
    transport = transport_ref(tenant_id=tenant_id, task_id="shared-conflict")
    first = admit(transport=transport, deps=process_a)
    process_a.execution_terminal.commit_terminal_outcome(
        tenant_id=first.identity.tenant_id,
        task_id=str(first.identity.task_id),
        run_id=first.identity.run_id,
        outcome=ExecutionTerminalOutcome.COMPLETED,
        reason="consumer_a",
    )

    process_b = fresh_admission_composition(admission_backing)
    with pytest.raises(ExecutionTerminalConflictError):
        process_b.execution_terminal.commit_terminal_outcome(
            tenant_id=first.identity.tenant_id,
            task_id=str(first.identity.task_id),
            run_id=first.identity.run_id,
            outcome=ExecutionTerminalOutcome.FAILED,
            reason="consumer_b",
        )


def test_a2_checkpoint_terminal_restart_converges_without_a3(
    admission_backing: DurableAdmissionBacking,
    tenant_id: str,
    tmp_path,
) -> None:
    db_path = tmp_path / "shared-terminal-a2.db"
    process_a, checkpoint_store = create_shared_terminal_composition(admission_backing, db_path)
    transport = transport_ref(tenant_id=tenant_id, task_id="shared-a2")
    first = admit(transport=transport, deps=process_a)
    a1 = first.identity.attempt_id
    checkpoint = _paused_checkpoint(
        tenant_id=tenant_id,
        task_id=str(first.identity.task_id),
        run_id=first.identity.run_id,
        attempt_id=a1,
    )
    checkpoint_store.save(checkpoint)

    a2 = process_a.attempt_lifecycle.transition_to_next_attempt(
        tenant_id=first.identity.tenant_id,
        run_id=first.identity.run_id,
        expected_attempt_id=a1,
        reason=AttemptTransitionReason.RETRY,
    ).active_attempt_id
    process_b = fresh_admission_composition(admission_backing)
    redelivery_a2 = admit(transport=transport, deps=process_b)
    assert redelivery_a2.identity.attempt_id == a2
    assert redelivery_a2.disposition is BackgroundExecutionReentryDisposition.EXECUTE

    process_b.execution_terminal.commit_terminal_outcome(
        tenant_id=first.identity.tenant_id,
        task_id=str(first.identity.task_id),
        run_id=first.identity.run_id,
        outcome=ExecutionTerminalOutcome.CANCELLED,
        reason="operator_cancel",
    )

    process_c = fresh_admission_composition(admission_backing)
    redelivery_terminal = admit(transport=transport, deps=process_c)
    assert redelivery_terminal.identity.attempt_id == a2
    assert (
        redelivery_terminal.disposition
        is BackgroundExecutionReentryDisposition.TERMINAL_ALREADY_RECORDED
    )

    _, resume_terminal = fresh_shared_checkpoint_composition(admission_backing, db_path)
    loaded = create_checkpoint_store(db_path).get_by_token(
        checkpoint.task_id,
        tenant_id,
        checkpoint.resume_token,
    )
    assert loaded is not None
    with pytest.raises(CheckpointNotResumableError):
        assert_checkpoint_resumable(loaded, execution_terminal=resume_terminal)


@dataclass
class _RecordingTerminalStore(ExecutionTerminalStore):
    """Custom durable terminal provider for pluginability proof."""

    backing: dict[tuple[str, str], ExecutionTerminalRecord] = field(default_factory=dict)

    @property
    def is_durable(self) -> bool:
        return True

    def load_record(self, *, tenant_id: str, task_id: str) -> ExecutionTerminalRecord | None:
        return self.backing.get((tenant_id, task_id))

    def put_if_absent(self, record: ExecutionTerminalRecord) -> bool:
        key = (record.tenant_id, record.task_id)
        if key in self.backing:
            return False
        self.backing[key] = record
        return True


def test_custom_terminal_store_can_back_all_consumers(
    admission_backing: DurableAdmissionBacking,
    tenant_id: str,
    tmp_path,
) -> None:
    custom_store = _RecordingTerminalStore()
    db_path = tmp_path / "custom-terminal.db"
    process_a = create_admission_dependencies(
        admission_backing,
        execution_terminal_store=custom_store,
    )
    checkpoint_store = create_checkpoint_store(db_path)
    transport = transport_ref(tenant_id=tenant_id, task_id="custom-terminal")
    first = admit(transport=transport, deps=process_a)
    checkpoint = _paused_checkpoint(
        tenant_id=tenant_id,
        task_id=str(first.identity.task_id),
        run_id=first.identity.run_id,
        attempt_id=first.identity.attempt_id,
    )
    checkpoint_store.save(checkpoint)

    process_a.execution_terminal.commit_terminal_outcome(
        tenant_id=first.identity.tenant_id,
        task_id=str(first.identity.task_id),
        run_id=first.identity.run_id,
        outcome=ExecutionTerminalOutcome.CANCELLED,
        reason="custom_provider",
    )

    process_b = fresh_admission_composition(
        admission_backing,
        execution_terminal_store=custom_store,
    )
    resume_terminal = process_b.execution_terminal
    redelivery = admit(transport=transport, deps=process_b)
    loaded = create_checkpoint_store(db_path).get_by_token(
        checkpoint.task_id,
        tenant_id,
        checkpoint.resume_token,
    )
    assert loaded is not None
    assert redelivery.disposition is BackgroundExecutionReentryDisposition.TERMINAL_ALREADY_RECORDED
    with pytest.raises(CheckpointNotResumableError):
        assert_checkpoint_resumable(loaded, execution_terminal=resume_terminal)


def test_integrated_lifecycle_a1_a2_terminal_restart_denies_execution(
    admission_backing: DurableAdmissionBacking,
    tenant_id: str,
    tmp_path,
) -> None:
    test_a2_checkpoint_terminal_restart_converges_without_a3(
        admission_backing,
        tenant_id,
        tmp_path,
    )

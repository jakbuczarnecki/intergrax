# © Artur Czarnecki. All rights reserved.

"""Integrated P0C lifecycle coherence proof across restart boundaries."""

from __future__ import annotations

import pytest

from intergrax.contracts.attempt_lifecycle import AttemptTransitionReason
from intergrax.contracts.execution_identity import mint_execution_id
from intergrax.contracts.execution_terminal import ExecutionTerminalOutcome
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
)
from tests.conformance.runtime.durability.restart import (
    fresh_admission_composition,
    fresh_checkpoint_composition,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_integrated_lifecycle_a1_a2_terminal_restart_denies_execution(
    admission_backing: DurableAdmissionBacking,
    tenant_id: str,
    tmp_path,
) -> None:
    # Phase 1 — first delivery establishes A1 and durable checkpoint snapshot.
    process_a = create_admission_dependencies(admission_backing)
    transport = transport_ref(tenant_id=tenant_id, task_id="integrated-lifecycle")
    first = admit(transport=transport, deps=process_a)
    a1 = first.identity.attempt_id
    db_path = tmp_path / "integrated-lifecycle.db"
    checkpoint_store = create_checkpoint_store(db_path)
    checkpoint = TaskCheckpoint(
        task_id=str(first.identity.task_id),
        tenant_id=tenant_id,
        resume_token="rt-integrated",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=Task(
            task_id=str(first.identity.task_id),
            tenant_id=tenant_id,
            user_id="user",
            message="paused",
            state=TaskState.WAITING_FOR_HUMAN,
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(enabled=True, resume_token="rt-integrated"),
            ),
        ).model_dump(mode="json"),
        runtime=minimal_runtime_checkpoint(
            task_id=str(first.identity.task_id),
            run_id=first.identity.run_id,
            attempt_id=a1,
            root_execution_id=mint_execution_id(),
        ),
    )
    checkpoint_store.save(checkpoint)

    # Phase 2 — explicit retry transition to A2, then process restart.
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

    # Phase 3 — terminal CANCELLED, second restart, redelivery denied.
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

    # Phase 4 — stale checkpoint resume denied by checkpoint-backed terminal authority.
    _, checkpoint_terminal = fresh_checkpoint_composition(db_path)
    checkpoint_terminal.commit_terminal_outcome(
        tenant_id=first.identity.tenant_id,
        task_id=str(first.identity.task_id),
        run_id=first.identity.run_id,
        outcome=ExecutionTerminalOutcome.CANCELLED,
        reason="operator_cancel",
    )
    loaded = create_checkpoint_store(db_path).get_by_token(
        checkpoint.task_id,
        tenant_id,
        checkpoint.resume_token,
    )
    assert loaded is not None
    with pytest.raises(CheckpointNotResumableError):
        assert_checkpoint_resumable(loaded, execution_terminal=checkpoint_terminal)

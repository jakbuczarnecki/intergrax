# © Artur Czarnecki. All rights reserved.

"""EE-B3-A — retry and recovery security gate."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_retry import (
    ExecutionFailureKind,
    ExecutionRetryAction,
    ExecutionRetryEligibilityRequest,
)
from intergrax.runtime.execution.attempt_lifecycle import (
    AttemptLifecycleService,
    InMemoryAttemptLifecycleStore,
)
from intergrax.runtime.execution.retry import (
    ExecutionAttemptRetryService,
    classify_execution_failure,
)
from intergrax.runtime.execution.retry.policy import (
    evaluate_execution_retry_eligibility,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.runtime.long_running.checkpoint_resume_validation import (
    CheckpointResumeEligibility,
    validate_checkpoint_authority_expansion,
)
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    minimal_runtime_checkpoint,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_contract import (
    TaskExecutionOptions,
    TaskLongRunningOptions,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _retry_request() -> ExecutionRetryEligibilityRequest:
    return ExecutionRetryEligibilityRequest(
        classification=classify_execution_failure(
            kind=ExecutionFailureKind.RETRYABLE_TRANSIENT
        ),
        attempt_number=1,
        max_attempts=3,
    )


def test_ee_b3_a_retry_preserves_run_id_mints_new_attempt() -> None:
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    service = ExecutionAttemptRetryService(lifecycle)
    tenant = "tenant-b3a"
    run_id = mint_run_id()
    attempt_a = mint_attempt_id()
    lifecycle.record_initial_attempt(
        tenant_id=tenant, run_id=run_id, attempt_id=attempt_a
    )
    transition = service.transition_for_retry(
        tenant_id=tenant,
        task_id=mint_task_id(),
        run_id=run_id,
        expected_attempt_id=attempt_a,
        request=_retry_request(),
    )
    assert transition is not None
    assert transition.run_id == run_id
    assert transition.active_attempt_id != attempt_a


def test_ee_b3_a_wrong_tenant_retry_does_not_read_other_tenant_lifecycle() -> None:
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    service = ExecutionAttemptRetryService(lifecycle)
    run_id = mint_run_id()
    attempt_a = mint_attempt_id()
    lifecycle.record_initial_attempt(
        tenant_id="tenant-a", run_id=run_id, attempt_id=attempt_a
    )
    transition = service.transition_for_retry(
        tenant_id="tenant-b",
        task_id=mint_task_id(),
        run_id=run_id,
        expected_attempt_id=attempt_a,
        request=_retry_request(),
    )
    assert transition is not None
    assert (
        lifecycle.get_active_attempt_id(tenant_id="tenant-a", run_id=run_id)
        == attempt_a
    )


def test_ee_b3_a_resume_rejects_authority_expansion() -> None:
    task_id = mint_task_id()
    task = Task(
        task_id=task_id,
        tenant_id="tenant-b3a",
        user_id="user",
        message="paused",
        state=TaskState.WAITING_FOR_HUMAN,
        execution_authority=ParentExecutionAuthority.unrestricted_root(),
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, resume_token="rt-b3a"),
        ),
    )
    checkpoint = TaskCheckpoint(
        task_id=task_id,
        tenant_id="tenant-b3a",
        resume_token="rt-b3a",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=task.model_dump(mode="json"),
        created_at_utc="2026-09-09T12:00:00+00:00",
        runtime=minimal_runtime_checkpoint(
            task_id=task_id,
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            root_execution_id=mint_execution_id(),
        ),
    )
    result = validate_checkpoint_authority_expansion(
        checkpoint,
        ParentExecutionAuthority.scoped(("read",)),
    )
    assert result.eligibility is CheckpointResumeEligibility.REJECT_AUTHORITY


def test_ee_b3_a_retry_eligibility_fail_closed_on_exhausted() -> None:
    result = evaluate_execution_retry_eligibility(
        ExecutionRetryEligibilityRequest(
            classification=classify_execution_failure(
                kind=ExecutionFailureKind.RETRYABLE_TRANSIENT
            ),
            attempt_number=3,
            max_attempts=3,
        ),
    )
    assert result.action is ExecutionRetryAction.FAIL

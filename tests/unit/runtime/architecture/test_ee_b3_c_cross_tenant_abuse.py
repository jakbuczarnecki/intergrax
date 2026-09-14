# © Artur Czarnecki. All rights reserved.

"""EE-B3-C — AC-02 cross-tenant resume and identity binding abuse."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.long_running.checkpoint_resume_validation import (
    CheckpointResumeEligibility,
    evaluate_checkpoint_resume_eligibility,
    validate_checkpoint_identity_binding,
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


def _paused_checkpoint(tenant_id: str) -> TaskCheckpoint:
    task_id = mint_task_id()
    task = Task(
        task_id=task_id,
        tenant_id=tenant_id,
        user_id="user",
        message="paused",
        state=TaskState.WAITING_FOR_HUMAN,
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, resume_token="rt-b3c"),
        ),
    )
    return TaskCheckpoint(
        task_id=task_id,
        tenant_id=tenant_id,
        resume_token="rt-b3c",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=task.model_dump(mode="json"),
        created_at_utc="2026-09-14T12:00:00+00:00",
        runtime=minimal_runtime_checkpoint(
            task_id=task_id,
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            root_execution_id=mint_execution_id(),
        ),
    )


def test_ee_b3_c_resume_rejects_cross_tenant_no_switch() -> None:
    checkpoint = _paused_checkpoint("tenant-a")
    result = evaluate_checkpoint_resume_eligibility(
        checkpoint,
        target_task_id=checkpoint.task_id,
        target_tenant_id="tenant-b",
    )
    assert result.eligibility is CheckpointResumeEligibility.REJECT_TENANT
    assert checkpoint.tenant_id == "tenant-a"


def test_ee_b3_c_identity_binding_rejects_cross_tenant() -> None:
    checkpoint = _paused_checkpoint("tenant-a")
    result = validate_checkpoint_identity_binding(
        checkpoint,
        target_task_id=checkpoint.task_id,
        target_tenant_id="tenant-b",
    )
    assert result.eligibility is CheckpointResumeEligibility.REJECT_TENANT


def test_ee_b3_c_identity_binding_rejects_forged_run_id() -> None:
    checkpoint = _paused_checkpoint("tenant-a")
    other_run = mint_run_id()
    result = validate_checkpoint_identity_binding(
        checkpoint,
        target_task_id=checkpoint.task_id,
        target_tenant_id="tenant-a",
        target_run_id=other_run,
    )
    assert result.eligibility is CheckpointResumeEligibility.REJECT_IDENTITY

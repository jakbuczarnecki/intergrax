# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R3 adversarial probes (production contracts; fixture/unit only)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from intergrax.agents.persistence.checkpoint_store import InMemoryAgentCheckpointStore, build_checkpoint
from intergrax.contracts.orchestration_topology import (
    OrchestrationSlotId,
    OrchestrationTopologyExecutionId,
)
from intergrax.contracts.partial_recovery import PartialRecoveryReason, PartialRecoveryRequest
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.partial_recovery import PartialRecoveryErrorCode
from intergrax.runtime.execution.fan_out_partial_recovery import (
    FanOutPartialRecoveryService,
    PartialRecoveryError,
)
from intergrax.runtime.long_running.checkpoint_resume_validation import (
    CheckpointResumeEligibility,
    evaluate_checkpoint_resume_eligibility,
    validate_checkpoint_identity_binding,
)
from intergrax.runtime.long_running.execution_tree_checkpoint import minimal_runtime_checkpoint
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions


def _paused_checkpoint(tenant_id: str) -> TaskCheckpoint:
    task_id = mint_task_id()
    task = Task(
        task_id=task_id,
        tenant_id=tenant_id,
        user_id="user",
        message="paused",
        state=TaskState.WAITING_FOR_HUMAN,
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, resume_token="rt-r3"),
        ),
    )
    return TaskCheckpoint(
        task_id=task_id,
        tenant_id=tenant_id,
        resume_token="rt-r3",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=task.model_dump(mode="json"),
        created_at_utc="2026-10-12T12:00:00+00:00",
        runtime=minimal_runtime_checkpoint(
            task_id=task_id,
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            root_execution_id=mint_execution_id(),
        ),
    )


def assert_checkpoint_recovery_cross_tenant_rejects() -> None:
    checkpoint = _paused_checkpoint("tenant-a")
    result = validate_checkpoint_identity_binding(
        checkpoint,
        target_task_id=checkpoint.task_id,
        target_tenant_id="tenant-b",
    )
    assert result.eligibility is CheckpointResumeEligibility.REJECT_TENANT
    eligibility = evaluate_checkpoint_resume_eligibility(
        checkpoint,
        target_task_id=checkpoint.task_id,
        target_tenant_id="tenant-b",
    )
    assert eligibility.eligibility is CheckpointResumeEligibility.REJECT_TENANT


def assert_missing_tenant_resume_rejects_no_global() -> None:
    checkpoint = _paused_checkpoint("tenant-a")
    result = validate_checkpoint_identity_binding(
        checkpoint,
        target_task_id=checkpoint.task_id,
        target_tenant_id="   ",
    )
    assert result.eligibility is CheckpointResumeEligibility.REJECT_TENANT


def assert_retry_resume_preserves_tenant_a() -> None:
    checkpoint = _paused_checkpoint("tenant-a")
    same = evaluate_checkpoint_resume_eligibility(
        checkpoint,
        target_task_id=checkpoint.task_id,
        target_tenant_id="tenant-a",
    )
    assert same.eligibility is not CheckpointResumeEligibility.REJECT_TENANT
    assert checkpoint.tenant_id == "tenant-a"


def assert_tenant_b_cannot_load_tenant_a_durable_checkpoint(tmp_db: Path | None = None) -> None:
    db_path = tmp_db
    if db_path is None:
        db_path = Path(".tmp/session/tenant-x-r3-fast/task_cp.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
    store = SQLiteTaskCheckpointStore(db_path=db_path)
    checkpoint = _paused_checkpoint("tenant-a")
    store.save(checkpoint)
    loaded_wrong = store.get_by_token(checkpoint.task_id, "tenant-b", checkpoint.resume_token)
    assert loaded_wrong is None
    loaded_right = store.get_by_token(checkpoint.task_id, "tenant-a", checkpoint.resume_token)
    assert loaded_right is not None
    assert loaded_right.tenant_id == "tenant-a"


def assert_agent_checkpoint_cross_tenant_load_rejects() -> None:
    store = InMemoryAgentCheckpointStore()
    run_id = "run-r3-a"
    cp = build_checkpoint(
        run_id=run_id,
        tenant_id="tenant-a",
        agent_id="agent-1",
        step_index=0,
        state_root={},
        side_effect_ledger=[],
        trace_step_count=0,
    )
    store.save(cp)
    assert store.get_latest(run_id, "tenant-b") is None
    assert store.get_latest(run_id, "tenant-a") is not None


def assert_partial_recovery_tenant_mismatch_rejects() -> None:
    checkpoint = _paused_checkpoint("tenant-a")
    service = FanOutPartialRecoveryService(
        adapter=Mock(),
        submission_port=Mock(),
        checkpoint_store=Mock(),
        recovery_admission=None,
    )
    request = PartialRecoveryRequest(
        root_execution_id=mint_execution_id(),
        topology_execution_id=OrchestrationTopologyExecutionId("topo-r3"),
        slot_id=OrchestrationSlotId("slot-1"),
        source_checkpoint_revision=checkpoint.revision or 1,
        source_attempt_id=mint_attempt_id(),
        recovery_reason=PartialRecoveryReason.TRANSIENT_SLOT_FAILURE,
    )
    with pytest.raises(PartialRecoveryError) as exc_info:
        service.validate_recovery_request(
            request,
            fan_out_request=Mock(),
            checkpoint=checkpoint,
            root_execution_id=mint_execution_id(),
            tenant_id="tenant-b",
        )
    assert exc_info.value.code == PartialRecoveryErrorCode.TENANT_MISMATCH

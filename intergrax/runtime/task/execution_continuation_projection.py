# © Artur Czarnecki. All rights reserved.

"""GR-5-R3 — default Task projection from canonical continuation snapshots."""

from __future__ import annotations

from datetime import datetime, timezone

from intergrax.contracts.agent_decision import HumanRequest
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationLifecycleState,
    ExecutionHumanVerdict,
    PendingExecutionContinuation,
)
from intergrax.contracts.execution_continuation_projection import (
    ExecutionContinuationProjectionError,
    ExecutionContinuationProjectionResult,
    ExecutionContinuationProjectionSink,
    ExecutionContinuationProjectionStatus,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import (
    TaskPauseRecord,
    VERDICT_APPROVE,
    VERDICT_ESCALATE,
    VERDICT_REJECT,
)


def governance_paused_for_lifecycle(
    state: ExecutionContinuationLifecycleState,
) -> bool:
    """Task ``governance.paused`` projection — not canonical lifecycle authority."""
    if state is ExecutionContinuationLifecycleState.PAUSE_REQUESTED:
        return False
    if state is ExecutionContinuationLifecycleState.RESUMED:
        return False
    if state in {
        ExecutionContinuationLifecycleState.REJECTED,
        ExecutionContinuationLifecycleState.CANCELLED,
    }:
        return False
    if state in {
        ExecutionContinuationLifecycleState.PAUSED,
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
        ExecutionContinuationLifecycleState.ESCALATED,
    }:
        return True
    return False


def _human_verdict_for_options(
    pending: PendingExecutionContinuation,
) -> str | None:
    if pending.lifecycle_state is ExecutionContinuationLifecycleState.RESUMED:
        if pending.human_verdict is ExecutionHumanVerdict.APPROVE:
            return VERDICT_APPROVE
        return None
    if pending.lifecycle_state is ExecutionContinuationLifecycleState.REJECTED:
        return VERDICT_REJECT
    if pending.lifecycle_state is ExecutionContinuationLifecycleState.ESCALATED:
        return VERDICT_ESCALATE
    if pending.lifecycle_state is ExecutionContinuationLifecycleState.RESUME_AUTHORIZED:
        if pending.human_verdict is ExecutionHumanVerdict.APPROVE:
            return VERDICT_APPROVE
    return None


def apply_canonical_projection_fields(
    task: Task,
    pending: PendingExecutionContinuation,
) -> None:
    """Mutate Task governance/options from one canonical snapshot (idempotent per revision)."""
    gov = task.runtime.governance
    state = pending.lifecycle_state
    gov.projected_continuation_id = pending.continuation_id
    gov.projected_continuation_revision = pending.revision
    gov.projected_continuation_lifecycle_state = state.value

    gov.paused = governance_paused_for_lifecycle(state)

    pause_id = pending.pause_id
    human_request_id = pending.human_request_id
    if state in {
        ExecutionContinuationLifecycleState.PAUSED,
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
        ExecutionContinuationLifecycleState.ESCALATED,
    }:
        if pause_id is None or human_request_id is None:
            raise ExecutionContinuationProjectionError(
                "canonical WAITING/PAUSED projection requires pause_id and human_request_id",
            )
        gov.pause_record = TaskPauseRecord(
            pause_id=pause_id,
            task_id=str(pending.identity.task_id),
            human_request_id=human_request_id,
            reason=pending.reason.value,
            created_at=pending.requested_at or datetime.now(timezone.utc).isoformat(),
        )
        if gov.human_request is None or gov.human_request.request_id != human_request_id:
            gov.human_request = HumanRequest(
                request_id=human_request_id,
                prompt=gov.human_request.prompt if gov.human_request else "",
                options=(
                    gov.human_request.options
                    if gov.human_request
                    else [
                        HumanResponseVerdict.APPROVE.value,
                        HumanResponseVerdict.REJECT.value,
                        HumanResponseVerdict.ESCALATE.value,
                    ]
                ),
                governed_continuation=pending.governed_correlation,
            )
    elif state in {
        ExecutionContinuationLifecycleState.RESUMED,
        ExecutionContinuationLifecycleState.REJECTED,
        ExecutionContinuationLifecycleState.CANCELLED,
    }:
        if state is ExecutionContinuationLifecycleState.RESUMED:
            gov.pause_record = None
            gov.declarative_hitl_pending = None
        if state is ExecutionContinuationLifecycleState.CANCELLED:
            gov.pause_record = None

    if state in {
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        ExecutionContinuationLifecycleState.PAUSED,
        ExecutionContinuationLifecycleState.PAUSE_REQUESTED,
    }:
        gov.hitl_resolution = None
        gov.governed_continuation_grant = None
        gov.physical_delegation_continuation_grant = None

    verdict_option = _human_verdict_for_options(pending)
    task.options.human.verdict = verdict_option
    if pause_id is not None:
        task.options.human.pause_id = pause_id
    if human_request_id is not None:
        task.options.human.human_request_id = human_request_id

    task.sync_metadata()


class TaskExecutionContinuationProjectionSink:
    """Default Task adapter implementing :class:`ExecutionContinuationProjectionSink`."""

    __slots__ = ("_task",)

    def __init__(self, task: Task) -> None:
        self._task = task

    def project(
        self,
        pending: PendingExecutionContinuation,
    ) -> ExecutionContinuationProjectionResult:
        task = self._task
        if str(pending.identity.task_id) != task.task_id:
            return ExecutionContinuationProjectionResult(
                status=ExecutionContinuationProjectionStatus.TASK_IDENTITY_MISMATCH,
            )
        gov = task.runtime.governance
        last_revision = gov.projected_continuation_revision
        last_id = gov.projected_continuation_id
        if (
            last_revision is not None
            and last_id == pending.continuation_id
            and pending.revision < last_revision
        ):
            return ExecutionContinuationProjectionResult(
                status=ExecutionContinuationProjectionStatus.STALE_IGNORED,
                applied_revision=last_revision,
            )
        try:
            apply_canonical_projection_fields(task, pending)
        except ExecutionContinuationProjectionError:
            raise
        except Exception as exc:
            raise ExecutionContinuationProjectionError(str(exc)) from exc
        return ExecutionContinuationProjectionResult(
            status=ExecutionContinuationProjectionStatus.APPLIED,
            applied_revision=pending.revision,
        )


def wire_task_execution_continuation_projection_sink(
    task: Task,
) -> TaskExecutionContinuationProjectionSink:
    """Explicit composition helper — no module singleton."""
    return TaskExecutionContinuationProjectionSink(task)


__all__ = [
    "TaskExecutionContinuationProjectionSink",
    "apply_canonical_projection_fields",
    "governance_paused_for_lifecycle",
    "wire_task_execution_continuation_projection_sink",
]

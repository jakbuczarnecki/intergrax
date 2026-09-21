# © Artur Czarnecki. All rights reserved.

"""GR-5-R3 — default Task projection from canonical continuation snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from intergrax.contracts.agent_decision import HumanRequest, HumanRequestUrgency
from intergrax.contracts.canonical_payload_hash import stable_payload_hash
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationLifecycleState,
    ExecutionHumanVerdict,
    PendingExecutionContinuation,
)
from intergrax.contracts.execution_continuation_projection import (
    ExecutionContinuationProjectionError,
    ExecutionContinuationProjectionResult,
    ExecutionContinuationProjectionStatus,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.request_contract import HumanTimeoutCoordinator
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import (
    HumanApprovalResolution,
    TaskPauseRecord,
    VERDICT_APPROVE,
    VERDICT_ESCALATE,
    VERDICT_REJECT,
)

_PROJECTION_REPLACEMENT_ELIGIBLE: frozenset[ExecutionContinuationLifecycleState] = (
    frozenset(
        {
            ExecutionContinuationLifecycleState.RESUMED,
            ExecutionContinuationLifecycleState.REJECTED,
            ExecutionContinuationLifecycleState.ESCALATED,
            ExecutionContinuationLifecycleState.CANCELLED,
        }
    )
)


def continuation_projection_allows_replacement(
    projected_lifecycle_state: str | None,
) -> bool:
    """Whether Task materialized view may switch to a new ``continuation_id``.

    Uses projection metadata only — not canonical lifecycle authority.
    """
    if projected_lifecycle_state is None:
        return True
    try:
        state = ExecutionContinuationLifecycleState(projected_lifecycle_state)
    except ValueError as exc:
        raise ExecutionContinuationProjectionError(
            f"unknown projected continuation lifecycle state: {projected_lifecycle_state!r}",
        ) from exc
    return state in _PROJECTION_REPLACEMENT_ELIGIBLE


def execution_continuation_projection_payload_digest(
    pending: PendingExecutionContinuation,
) -> str:
    """Deterministic fingerprint of canonical fields that drive Task/Human projection."""
    identity = pending.identity
    governed: dict[str, Any] | None = None
    if pending.governed_correlation is not None:
        governed = pending.governed_correlation.model_dump(mode="json")
    payload: dict[str, Any] = {
        "continuation_id": pending.continuation_id,
        "identity": {
            "task_id": str(identity.task_id),
            "run_id": str(identity.run_id),
            "attempt_id": str(identity.attempt_id),
            "execution_id": str(identity.execution_id),
        },
        "lifecycle_state": pending.lifecycle_state.value,
        "revision": pending.revision,
        "reason": pending.reason.value,
        "human_verdict": (
            pending.human_verdict.value if pending.human_verdict is not None else None
        ),
        "pause_id": pending.pause_id,
        "human_request_id": pending.human_request_id,
        "requested_at": pending.requested_at,
        "governed_correlation": governed,
    }
    return stable_payload_hash(payload)


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
    raise ExecutionContinuationProjectionError(
        f"unknown continuation lifecycle state for pause projection: {state!s}",
    )


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


def _validate_pending_snapshot_for_projection(
    pending: PendingExecutionContinuation,
) -> None:
    state = pending.lifecycle_state
    if state in {
        ExecutionContinuationLifecycleState.PAUSED,
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
        ExecutionContinuationLifecycleState.ESCALATED,
    }:
        if pending.pause_id is None or pending.human_request_id is None:
            raise ExecutionContinuationProjectionError(
                "canonical WAITING/PAUSED projection requires pause_id and human_request_id",
            )


def _same_revision_payload_conflict(
    task: Task,
    pending: PendingExecutionContinuation,
    incoming_digest: str,
) -> None:
    gov = task.runtime.governance
    stored_digest = gov.projected_continuation_payload_digest
    if stored_digest is not None:
        if stored_digest != incoming_digest:
            raise ExecutionContinuationProjectionError(
                "revision payload conflict for same continuation revision",
            )
        return
    projected_state = gov.projected_continuation_lifecycle_state
    if projected_state is not None and projected_state != pending.lifecycle_state.value:
        raise ExecutionContinuationProjectionError(
            "revision payload conflict for same continuation revision",
        )
    record = gov.pause_record
    if (
        record is not None
        and pending.pause_id is not None
        and record.pause_id != pending.pause_id
    ):
        raise ExecutionContinuationProjectionError(
            "revision payload conflict for same continuation revision",
        )
    if (
        record is not None
        and pending.human_request_id is not None
        and record.human_request_id != pending.human_request_id
    ):
        raise ExecutionContinuationProjectionError(
            "revision payload conflict for same continuation revision",
        )


@dataclass(frozen=True, slots=True)
class _PreparedTaskContinuationProjection:
    continuation_id: str
    revision: int
    lifecycle_state_value: str
    payload_digest: str
    paused: bool
    pause_record: TaskPauseRecord | None
    human_request: HumanRequest | None
    clear_hitl_and_grants: bool
    hitl_resolution: HumanApprovalResolution | None
    human_verdict_option: str | None
    human_pause_id: str | None
    human_request_id_option: str | None
    clear_pause_on_resume: bool
    clear_pause_on_cancel: bool
    clear_declarative_hitl_on_resume: bool
    replace_completed_continuation: bool
    idempotent_no_op: bool


def prepare_task_continuation_projection(
    task: Task,
    pending: PendingExecutionContinuation,
    *,
    accepted_hitl_resolution: HumanApprovalResolution | None = None,
) -> _PreparedTaskContinuationProjection:
    """Validate and derive a complete Task projection — does not mutate ``task``."""
    if str(pending.identity.task_id) != task.task_id:
        raise ExecutionContinuationProjectionError(
            "task identity mismatch for projection"
        )

    incoming_digest = execution_continuation_projection_payload_digest(pending)
    gov = task.runtime.governance
    last_revision = gov.projected_continuation_revision
    last_id = gov.projected_continuation_id
    projected_state = gov.projected_continuation_lifecycle_state
    replace_completed = False

    if last_id is not None and last_id != pending.continuation_id:
        if not continuation_projection_allows_replacement(projected_state):
            raise ExecutionContinuationProjectionError(
                "task already projects a different continuation_id",
            )
        replace_completed = True
    elif (
        last_revision is not None
        and last_id == pending.continuation_id
        and pending.revision == last_revision
    ):
        _same_revision_payload_conflict(task, pending, incoming_digest)
        stored_digest = gov.projected_continuation_payload_digest
        if stored_digest is not None and stored_digest == incoming_digest:
            return _PreparedTaskContinuationProjection(
                continuation_id=pending.continuation_id,
                revision=pending.revision,
                lifecycle_state_value=pending.lifecycle_state.value,
                payload_digest=incoming_digest,
                paused=governance_paused_for_lifecycle(pending.lifecycle_state),
                pause_record=None,
                human_request=None,
                clear_hitl_and_grants=False,
                hitl_resolution=None,
                human_verdict_option=None,
                human_pause_id=None,
                human_request_id_option=None,
                clear_pause_on_resume=False,
                clear_pause_on_cancel=False,
                clear_declarative_hitl_on_resume=False,
                replace_completed_continuation=False,
                idempotent_no_op=True,
            )
        if stored_digest is None:
            raise ExecutionContinuationProjectionError(
                "legacy projection lacks payload digest; same-revision equality cannot be verified",
            )

    _validate_pending_snapshot_for_projection(pending)
    state = pending.lifecycle_state
    paused = governance_paused_for_lifecycle(state)

    pause_id = pending.pause_id
    human_request_id = pending.human_request_id
    pause_record: TaskPauseRecord | None = None
    human_request: HumanRequest | None = None
    clear_pause_on_resume = False
    clear_pause_on_cancel = False
    clear_declarative_hitl_on_resume = False

    if state in {
        ExecutionContinuationLifecycleState.PAUSED,
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
        ExecutionContinuationLifecycleState.ESCALATED,
    }:
        assert pause_id is not None and human_request_id is not None
        pause_record = TaskPauseRecord(
            pause_id=pause_id,
            task_id=str(pending.identity.task_id),
            human_request_id=human_request_id,
            reason=pending.reason.value,
            created_at=pending.requested_at or datetime.now(timezone.utc).isoformat(),
        )
        existing = gov.human_request
        if existing is not None and existing.request_id == human_request_id:
            human_request = existing.model_copy(
                update={"governed_continuation": pending.governed_correlation},
            )
        else:
            human_request = HumanRequest(
                request_id=human_request_id,
                prompt=existing.prompt if existing else "",
                options=(
                    existing.options
                    if existing
                    else [
                        HumanResponseVerdict.APPROVE.value,
                        HumanResponseVerdict.REJECT.value,
                        HumanResponseVerdict.ESCALATE.value,
                    ]
                ),
                urgency=existing.urgency if existing else HumanRequestUrgency.NORMAL,
                timeout_seconds=existing.timeout_seconds if existing else None,
                default_on_timeout=existing.default_on_timeout if existing else None,
                context_artifacts=list(existing.context_artifacts) if existing else [],
                governed_continuation=pending.governed_correlation,
            )
    elif state is ExecutionContinuationLifecycleState.RESUMED:
        clear_pause_on_resume = True
        clear_declarative_hitl_on_resume = gov.declarative_hitl_pending is None
    elif state is ExecutionContinuationLifecycleState.CANCELLED:
        clear_pause_on_cancel = True
    elif state in {
        ExecutionContinuationLifecycleState.PAUSE_REQUESTED,
        ExecutionContinuationLifecycleState.REJECTED,
    }:
        pass
    else:
        raise ExecutionContinuationProjectionError(
            f"unhandled continuation lifecycle state: {state!s}",
        )

    clear_hitl = state in {
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        ExecutionContinuationLifecycleState.PAUSED,
        ExecutionContinuationLifecycleState.PAUSE_REQUESTED,
    }

    hitl_resolution: HumanApprovalResolution | None = None
    if accepted_hitl_resolution is not None:
        if state in {
            ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
            ExecutionContinuationLifecycleState.REJECTED,
            ExecutionContinuationLifecycleState.ESCALATED,
            ExecutionContinuationLifecycleState.RESUMED,
        }:
            hitl_resolution = accepted_hitl_resolution
    elif clear_hitl:
        hitl_resolution = None

    return _PreparedTaskContinuationProjection(
        continuation_id=pending.continuation_id,
        revision=pending.revision,
        lifecycle_state_value=state.value,
        payload_digest=incoming_digest,
        paused=paused,
        pause_record=pause_record,
        human_request=human_request,
        clear_hitl_and_grants=clear_hitl,
        hitl_resolution=hitl_resolution,
        human_verdict_option=_human_verdict_for_options(pending),
        human_pause_id=pause_id,
        human_request_id_option=human_request_id,
        clear_pause_on_resume=clear_pause_on_resume,
        clear_pause_on_cancel=clear_pause_on_cancel,
        clear_declarative_hitl_on_resume=clear_declarative_hitl_on_resume,
        replace_completed_continuation=replace_completed,
        idempotent_no_op=False,
    )


def _clear_episode_projection_residue(task: Task) -> None:
    gov = task.runtime.governance
    gov.human_request = None
    gov.pause_record = None
    gov.hitl_resolution = None
    gov.governed_continuation_grant = None
    gov.physical_delegation_continuation_grant = None
    human = task.options.human
    human.verdict = None
    human.pause_id = None
    human.human_request_id = None
    human.response_text = None


def commit_task_continuation_projection(
    task: Task,
    prepared: _PreparedTaskContinuationProjection,
) -> None:
    """Apply a validated prepared projection in one logical commit."""
    if prepared.idempotent_no_op:
        return

    gov = task.runtime.governance
    if prepared.replace_completed_continuation:
        _clear_episode_projection_residue(task)

    gov.projected_continuation_id = prepared.continuation_id
    gov.projected_continuation_revision = prepared.revision
    gov.projected_continuation_lifecycle_state = prepared.lifecycle_state_value
    gov.projected_continuation_payload_digest = prepared.payload_digest
    gov.paused = prepared.paused

    if prepared.clear_pause_on_resume or prepared.clear_pause_on_cancel:
        gov.pause_record = None
    elif prepared.pause_record is not None:
        gov.pause_record = prepared.pause_record

    if prepared.clear_declarative_hitl_on_resume:
        gov.declarative_hitl_pending = None

    if prepared.human_request is not None:
        gov.human_request = prepared.human_request
        if prepared.human_request.timeout_seconds is not None:
            HumanTimeoutCoordinator.attach_to_task(task, prepared.human_request)

    if prepared.clear_hitl_and_grants:
        gov.hitl_resolution = None
        gov.governed_continuation_grant = None
        gov.physical_delegation_continuation_grant = None
    elif prepared.hitl_resolution is not None:
        gov.hitl_resolution = prepared.hitl_resolution

    task.options.human.verdict = prepared.human_verdict_option
    if prepared.human_pause_id is not None:
        task.options.human.pause_id = prepared.human_pause_id
    if prepared.human_request_id_option is not None:
        task.options.human.human_request_id = prepared.human_request_id_option
    if prepared.hitl_resolution is not None and prepared.hitl_resolution.response_text:
        task.options.human.response_text = prepared.hitl_resolution.response_text

    task.sync_metadata()


def apply_canonical_projection_fields(
    task: Task,
    pending: PendingExecutionContinuation,
    *,
    accepted_hitl_resolution: HumanApprovalResolution | None = None,
) -> None:
    """Mutate Task governance/options from one canonical snapshot (idempotent per revision)."""
    prepared = prepare_task_continuation_projection(
        task,
        pending,
        accepted_hitl_resolution=accepted_hitl_resolution,
    )
    commit_task_continuation_projection(task, prepared)


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
    "commit_task_continuation_projection",
    "continuation_projection_allows_replacement",
    "execution_continuation_projection_payload_digest",
    "governance_paused_for_lifecycle",
    "prepare_task_continuation_projection",
    "wire_task_execution_continuation_projection_sink",
]

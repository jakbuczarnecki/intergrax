# © Artur Czarnecki. All rights reserved.

"""GR-5-R4 — canonical continuation helpers for private Nexus orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    ExecutionContinuationLookup,
    ExecutionContinuationPort,
    ExecutionContinuationResumeCommand,
    ExecutionPauseRequest,
    PendingExecutionContinuation,
)
from intergrax.contracts.execution_continuation_projection import (
    ExecutionContinuationProjectionSink,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)
from intergrax.contracts.governed_continuation import GovernedContinuationRequest
from intergrax.contracts.governed_continuation_correlation import (
    ContinuationReason,
    GovernedContinuationCorrelation,
)
from intergrax.runtime.execution.continuation.lifecycle_driver import (
    ExecutionContinuationLifecycleDriver,
)
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.task.task import Task

__all__ = [
    "InternalHitlContinuationCapabilityError",
    "InternalOrchestrationContinuation",
    "canonical_allows_planning_progress_after_human_gate",
    "canonical_execution_is_resumed",
    "canonical_resume_after_authorization",
    "establish_canonical_hitl_pause",
    "execution_continuation_identity_for_task",
    "require_internal_hitl_continuation",
    "resolve_continuation_id_for_execution",
]


class InternalHitlContinuationCapabilityError(RuntimeError):
    """Canonical HITL orchestration requires an injected continuation capability."""


@dataclass(frozen=True, slots=True)
class InternalOrchestrationContinuation:
    """Execution Engine–owned continuation capability for internal orchestration."""

    port: ExecutionContinuationPort
    lifecycle_driver: ExecutionContinuationLifecycleDriver
    projection_sink: ExecutionContinuationProjectionSink | None = None


def require_internal_hitl_continuation(
    capability: InternalOrchestrationContinuation | None,
) -> InternalOrchestrationContinuation:
    if capability is None:
        raise InternalHitlContinuationCapabilityError(
            "canonical execution continuation required for internal HITL orchestration",
        )
    if capability.port is None or capability.lifecycle_driver is None:
        raise InternalHitlContinuationCapabilityError(
            "canonical execution continuation port and lifecycle driver required",
        )
    return capability


def execution_continuation_identity_for_task(
    task: Task,
    *,
    run_id: str,
    attempt_id: str,
    execution_id: str,
) -> ExecutionContinuationIdentity:
    return _identity_from_parts(
        task_id=task.task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )


def _identity_from_parts(
    *,
    task_id: TaskId | str,
    run_id: RunId | str,
    attempt_id: AttemptId | str,
    execution_id: ExecutionId | str,
) -> ExecutionContinuationIdentity:
    return ExecutionContinuationIdentity(
        task_id=TaskId(str(task_id)),
        run_id=RunId(str(run_id)),
        attempt_id=AttemptId(str(attempt_id)),
        execution_id=ExecutionId(str(execution_id)),
    )


def _load_pending_optional(
    port: ExecutionContinuationPort,
    *,
    continuation_id: str,
) -> PendingExecutionContinuation | None:
    try:
        return port.get_pending(ExecutionContinuationLookup(continuation_id=continuation_id))
    except ExecutionContinuationError as exc:
        if exc.code is ExecutionContinuationErrorCode.NOT_FOUND:
            return None
        raise


def resolve_continuation_id_for_execution(
    execution: AgentExecutionResult,
    *,
    governed_request: GovernedContinuationRequest | None = None,
) -> str:
    if governed_request is not None:
        return governed_request.continuation_request_id
    human_request = execution.human_request
    if human_request is not None and human_request.governed_continuation is not None:
        return human_request.governed_continuation.continuation_request_id
    if human_request is not None:
        return f"gcr_hr_{human_request.request_id}"
    raise InternalHitlContinuationCapabilityError(
        "human_request required to resolve continuation_id for graph HITL pause",
    )


def establish_canonical_hitl_pause(
    task: Task,
    *,
    identity: ExecutionContinuationIdentity,
    continuation_id: str,
    reason: ContinuationReason,
    pause_id: str,
    human_request_id: str,
    capability: InternalOrchestrationContinuation,
    governed_correlation: GovernedContinuationCorrelation | None = None,
    human_prompt: str | None = None,
    execution_interrupt: object | None = None,
) -> PendingExecutionContinuation:
    """Canonical pause lifecycle + Task projection — not Task-only authority."""
    port = capability.port
    driver = capability.lifecycle_driver
    resolved_governed = governed_correlation
    if resolved_governed is None:
        resolved_governed = GovernedContinuationCorrelation(
            continuation_request_id=continuation_id,
            reason=reason,
            task_id=identity.task_id,
            run_id=identity.run_id,
            attempt_id=identity.attempt_id,
            execution_id=identity.execution_id,
            operation_id=f"internal_hitl_{human_request_id}",
        )
    pending = _load_pending_optional(port, continuation_id=continuation_id)
    if pending is None:
        pending = port.request_pause(
            ExecutionPauseRequest(
                identity=identity,
                continuation_id=continuation_id,
                reason=reason,
                governed_correlation=resolved_governed,
                pause_id=pause_id,
                human_request_id=human_request_id,
                requested_at=datetime.now(timezone.utc).isoformat(),
            ),
        )
    if pending.lifecycle_state is ExecutionContinuationLifecycleState.PAUSE_REQUESTED:
        pending = driver.record_execution_reached_safe_pause(
            continuation_id,
            execution_pause_established=True,
        )
    if pending.lifecycle_state is ExecutionContinuationLifecycleState.PAUSED:
        pending = driver.record_ready_for_human_resolution(continuation_id)
    if pending.lifecycle_state not in {
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
        ExecutionContinuationLifecycleState.RESUMED,
        ExecutionContinuationLifecycleState.REJECTED,
        ExecutionContinuationLifecycleState.ESCALATED,
        ExecutionContinuationLifecycleState.CANCELLED,
    }:
        raise ExecutionContinuationError(
            f"unexpected continuation state after pause establishment: {pending.lifecycle_state}",
            code=ExecutionContinuationErrorCode.INVALID_TRANSITION,
        )
    HumanPauseCoordinator.project_continuation(
        task,
        pending,
        projection_sink=capability.projection_sink,
    )
    if human_prompt and task.runtime.governance.human_request is not None:
        task.runtime.governance.human_request = (
            task.runtime.governance.human_request.model_copy(update={"prompt": human_prompt})
        )
    if execution_interrupt is not None:
        task.runtime.governance.execution_interrupt = execution_interrupt  # type: ignore[assignment]
    task.sync_metadata()
    return pending


def canonical_execution_is_resumed(
    capability: InternalOrchestrationContinuation | None,
    *,
    identity: ExecutionContinuationIdentity,
) -> bool:
    if capability is None:
        return False
    try:
        pending = capability.port.get_pending(ExecutionContinuationLookup(identity=identity))
    except ExecutionContinuationError as exc:
        if exc.code is ExecutionContinuationErrorCode.NOT_FOUND:
            return False
        raise
    return pending.lifecycle_state is ExecutionContinuationLifecycleState.RESUMED


def canonical_allows_planning_progress_after_human_gate(
    capability: InternalOrchestrationContinuation | None,
    *,
    identity: ExecutionContinuationIdentity,
    human_approval_required: bool,
) -> bool:
    if not human_approval_required:
        return True
    hitl = require_internal_hitl_continuation(capability)
    try:
        pending = hitl.port.get_pending(ExecutionContinuationLookup(identity=identity))
    except ExecutionContinuationError as exc:
        if exc.code is ExecutionContinuationErrorCode.NOT_FOUND:
            return False
        raise
    return pending.lifecycle_state is ExecutionContinuationLifecycleState.RESUMED


def canonical_resume_after_authorization(
    task: Task,
    authorized: PendingExecutionContinuation,
    *,
    capability: InternalOrchestrationContinuation,
) -> PendingExecutionContinuation:
    resumed = capability.port.resume(
        ExecutionContinuationResumeCommand(
            continuation_id=authorized.continuation_id,
            identity=authorized.identity,
            expected_revision=authorized.revision,
        ),
    )
    HumanPauseCoordinator.project_continuation(
        task,
        resumed,
        projection_sink=capability.projection_sink,
    )
    task.sync_metadata()
    return resumed

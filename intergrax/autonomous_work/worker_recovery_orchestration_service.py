# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker recovery orchestration service (AW-6B).

Executes bounded AW-6A recovery decisions as durable episodes and resumes
original work on success. Does not classify obstacles or mint authority.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import TypeVar

from intergrax.autonomous_work.lifecycle import (
    AutonomousWorkInvalidLifecycleTransition,
    AutonomousWorkLifecycleStateConflict,
    WorkerLifecycleService,
    WorkerLifecycleTransitionRequest,
)
from intergrax.autonomous_work.recovery_orchestration_ports import (
    CanonicalExecutionOutcomeReader,
    CanonicalExecutionTerminalDisposition,
    HumanDecisionRequest,
    HumanDecisionRequestPort,
    PortAvailabilityDisposition,
    UnavailableCanonicalExecutionOutcomeReader,
    UnavailableHumanDecisionRequestPort,
    UnavailableWorkerCapabilityAcquisitionPort,
    UnavailableWorkerEscalationPort,
    UnavailableWorkerRecoveryReplanPort,
    WorkerCapabilityAcquisitionPort,
    WorkerCapabilityAcquisitionRequest,
    WorkerEscalationPort,
    WorkerEscalationRequest,
    WorkerRecoveryExecutionDispatchPort,
    WorkerRecoveryReplanPort,
    WorkerRecoveryReplanRequest,
)
from intergrax.autonomous_work.repository import (
    AutonomousWorkEntityNotFound,
    AutonomousWorkRevisionConflict,
    WorkerGoalRepository,
    WorkerInstanceRepository,
    WorkerRecoveryEpisodeClaimStatus,
    WorkerRecoveryEpisodeCreateStatus,
    WorkerRecoveryEpisodeRepository,
    WorkContinuityStateRepository,
)
from intergrax.contracts.autonomous_work.execution_dispatch import (
    WorkerExecutionDispatchDisposition,
    WorkerExecutionDispatchRequest,
    WorkerExecutionDispatchRejectionReason,
    WorkerExecutionSourceKind,
)
from intergrax.contracts.autonomous_work.goal import WorkerGoalStatus
from intergrax.contracts.autonomous_work.ids import WorkerInstanceId
from intergrax.contracts.autonomous_work.lifecycle import WorkerLifecycleState
from intergrax.contracts.autonomous_work.obstacle_recovery import RecoveryStrategy
from intergrax.contracts.autonomous_work.recovery_orchestration import (
    RecoveryEpisodeStatus,
    RecoveryExecutionBounds,
    WorkerOriginalWorkResumeDisposition,
    WorkerOriginalWorkResumeIntent,
    WorkerOriginalWorkResumeResult,
    WorkerRecoveryAttemptDisposition,
    WorkerRecoveryAttemptResult,
    WorkerRecoveryEpisode,
    WorkerRecoveryOrchestrationDisposition,
    WorkerRecoveryOrchestrationRequest,
    WorkerRecoveryOrchestrationResult,
    derive_recovery_attempt_id,
    derive_recovery_episode_id,
    is_terminal_recovery_episode_status,
)
from intergrax.contracts.autonomous_work.revision import Revision, initial_revision
from intergrax.contracts.autonomous_work.worker import WorkerInstance

_INELIGIBLE_WORKER_LIFECYCLE_STATES: frozenset[WorkerLifecycleState] = frozenset(
    {
        WorkerLifecycleState.PAUSED,
        WorkerLifecycleState.QUARANTINED,
        WorkerLifecycleState.STOPPED,
        WorkerLifecycleState.PROVISIONING,
    }
)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class WorkerRecoveryLifecycleTransitionDisposition(StrEnum):
    """Typed lifecycle transition outcome for recovery orchestration."""

    APPLIED = "APPLIED"
    UNCHANGED = "UNCHANGED"
    CONFLICT = "CONFLICT"
    NOT_FOUND = "NOT_FOUND"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class WorkerRecoveryLifecycleTransitionOutcome:
    """Lifecycle transition result consumed by recovery orchestration."""

    disposition: WorkerRecoveryLifecycleTransitionDisposition
    worker: WorkerInstance | None = None


def _utc_now() -> datetime:
    return datetime.now(UTC)


class WorkerRecoveryOrchestrationService[InputT, OutputT]:
    """Durable recovery episode orchestration with resume-original-work semantics."""

    def __init__(
        self,
        *,
        episode_repository: WorkerRecoveryEpisodeRepository,
        worker_instance_repository: WorkerInstanceRepository,
        worker_goal_repository: WorkerGoalRepository,
        continuity_repository: WorkContinuityStateRepository,
        lifecycle_service: WorkerLifecycleService,
        dispatch_port: WorkerRecoveryExecutionDispatchPort,
        replan_port: WorkerRecoveryReplanPort | None = None,
        capability_port: WorkerCapabilityAcquisitionPort | None = None,
        human_decision_port: HumanDecisionRequestPort | None = None,
        escalation_port: WorkerEscalationPort | None = None,
        execution_outcome_reader: CanonicalExecutionOutcomeReader | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._episode_repository = episode_repository
        self._worker_instance_repository = worker_instance_repository
        self._worker_goal_repository = worker_goal_repository
        self._continuity_repository = continuity_repository
        self._lifecycle_service = lifecycle_service
        self._dispatch_port = dispatch_port
        self._replan_port = replan_port or UnavailableWorkerRecoveryReplanPort()
        self._capability_port = capability_port or UnavailableWorkerCapabilityAcquisitionPort()
        self._human_decision_port = human_decision_port or UnavailableHumanDecisionRequestPort()
        self._escalation_port = escalation_port or UnavailableWorkerEscalationPort()
        self._execution_outcome_reader = (
            execution_outcome_reader or UnavailableCanonicalExecutionOutcomeReader()
        )
        self._clock = clock or _utc_now

    async def orchestrate(
        self,
        request: WorkerRecoveryOrchestrationRequest,
        *,
        dispatch_request: WorkerExecutionDispatchRequest[InputT, OutputT] | None = None,
        bounds: RecoveryExecutionBounds | None = None,
    ) -> WorkerRecoveryOrchestrationResult:
        now = self._clock()
        decision = request.decision
        episode_seed = _episode_from_request(request, started_at=now)
        create_result = self._episode_repository.create_or_get(episode_seed)
        if create_result.status is WorkerRecoveryEpisodeCreateStatus.CONFLICT:
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.CONFLICT,
                episode=create_result.episode,
            )

        episode = create_result.episode
        if is_terminal_recovery_episode_status(episode.status):
            return _terminal_reentry_result(episode)

        if episode.recovery_decision_id != decision.decision_id:
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.STALE_SOURCE,
                episode=episode,
            )
        if episode.decision_policy_version != decision.decision_policy_version:
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.STALE_SOURCE,
                episode=episode,
            )

        resolved_bounds = bounds or request.bounds
        if _deadline_exceeded(episode, resolved_bounds, now):
            episode = self._episode_repository.mark_escalated(
                recovery_episode_id=episode.recovery_episode_id,
                expected_revision=episode.revision,
                completed_at=now,
                terminal_reason="recovery_deadline_exceeded",
            )
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.LIMIT_EXCEEDED,
                episode=episode,
            )

        reconciled = self._reconcile_in_progress_episode(
            episode,
            request=request,
            now=now,
        )
        if reconciled is not None:
            return reconciled

        if _episode_requires_reconciliation(episode):
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.RECONCILIATION_REQUIRED,
                episode=episode,
            )

        strategy = decision.strategy
        if strategy is RecoveryStrategy.STOP:
            return self._handle_stop(episode, now=now)
        if strategy is RecoveryStrategy.ESCALATE:
            return self._handle_escalate(episode, now=now)
        if strategy is RecoveryStrategy.QUARANTINE:
            return self._handle_quarantine(episode, request=request, now=now)
        if strategy is RecoveryStrategy.REQUEST_HUMAN_DECISION:
            return self._handle_human_decision(episode, request=request, now=now)
        if strategy in {RecoveryStrategy.WAIT, RecoveryStrategy.THROTTLE}:
            return self._handle_wait(episode, decision=decision, now=now)
        if strategy in {RecoveryStrategy.ADAPT_INTEGRATION, RecoveryStrategy.ACQUIRE_CAPABILITY}:
            return self._handle_capability_deferred(episode, request=request, now=now)
        if strategy is RecoveryStrategy.REPLAN:
            return await self._handle_replan(
                episode,
                request=request,
                dispatch_request=dispatch_request,
                bounds=resolved_bounds,
                now=now,
            )
        if strategy is RecoveryStrategy.RETRY:
            return await self._handle_retry(
                episode,
                request=request,
                dispatch_request=dispatch_request,
                bounds=resolved_bounds,
                now=now,
            )
        episode = self._episode_repository.mark_escalated(
            recovery_episode_id=episode.recovery_episode_id,
            expected_revision=episode.revision,
            completed_at=now,
            terminal_reason="unsupported_recovery_strategy",
        )
        return WorkerRecoveryOrchestrationResult(
            disposition=WorkerRecoveryOrchestrationDisposition.ESCALATED,
            episode=episode,
        )

    def _handle_stop(
        self,
        episode: WorkerRecoveryEpisode,
        *,
        now: datetime,
    ) -> WorkerRecoveryOrchestrationResult:
        episode = self._episode_repository.mark_stopped(
            recovery_episode_id=episode.recovery_episode_id,
            expected_revision=episode.revision,
            completed_at=now,
            terminal_reason="policy_stop",
        )
        return WorkerRecoveryOrchestrationResult(
            disposition=WorkerRecoveryOrchestrationDisposition.STOPPED,
            episode=episode,
        )

    def _handle_escalate(
        self,
        episode: WorkerRecoveryEpisode,
        *,
        now: datetime,
    ) -> WorkerRecoveryOrchestrationResult:
        escalation = self._escalation_port.escalate(
            WorkerEscalationRequest(episode=episode, reason="recovery_escalate"),
        )
        episode = self._episode_repository.mark_escalated(
            recovery_episode_id=episode.recovery_episode_id,
            expected_revision=episode.revision,
            completed_at=now,
            terminal_reason="recovery_escalate",
        )
        if escalation.disposition is PortAvailabilityDisposition.UNAVAILABLE:
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.ESCALATED,
                episode=episode,
            )
        return WorkerRecoveryOrchestrationResult(
            disposition=WorkerRecoveryOrchestrationDisposition.ESCALATED,
            episode=episode,
        )

    def _handle_quarantine(
        self,
        episode: WorkerRecoveryEpisode,
        *,
        request: WorkerRecoveryOrchestrationRequest,
        now: datetime,
    ) -> WorkerRecoveryOrchestrationResult:
        worker = self._load_worker(episode.worker_instance_id)
        if worker is not None:
            transition = self._transition_lifecycle(
                worker=worker,
                target_state=WorkerLifecycleState.QUARANTINED,
                reason="recovery_quarantine",
            )
            if transition.disposition in {
                WorkerRecoveryLifecycleTransitionDisposition.CONFLICT,
                WorkerRecoveryLifecycleTransitionDisposition.NOT_FOUND,
                WorkerRecoveryLifecycleTransitionDisposition.INVALID,
            }:
                return WorkerRecoveryOrchestrationResult(
                    disposition=WorkerRecoveryOrchestrationDisposition.CONFLICT,
                    episode=episode,
                )
        episode = self._episode_repository.mark_quarantined(
            recovery_episode_id=episode.recovery_episode_id,
            expected_revision=episode.revision,
            completed_at=now,
            terminal_reason="suspicious_or_unsafe",
        )
        return WorkerRecoveryOrchestrationResult(
            disposition=WorkerRecoveryOrchestrationDisposition.QUARANTINED,
            episode=episode,
        )

    def _handle_human_decision(
        self,
        episode: WorkerRecoveryEpisode,
        *,
        request: WorkerRecoveryOrchestrationRequest,
        now: datetime,
    ) -> WorkerRecoveryOrchestrationResult:
        if episode.status is RecoveryEpisodeStatus.WAITING_FOR_HUMAN:
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.WAITING_FOR_HUMAN,
                episode=episode,
            )
        human_ref = request.decision.human_decision_ref
        if human_ref is None:
            episode = self._episode_repository.mark_escalated(
                recovery_episode_id=episode.recovery_episode_id,
                expected_revision=episode.revision,
                completed_at=now,
                terminal_reason="human_decision_ref_missing",
            )
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.ESCALATED,
                episode=episode,
            )
        hitl = self._human_decision_port.request_human_decision(
            HumanDecisionRequest(
                episode=episode,
                human_decision_ref=human_ref,
                requested_at=now,
            )
        )
        if hitl.disposition is PortAvailabilityDisposition.UNAVAILABLE:
            episode = self._episode_repository.mark_escalated(
                recovery_episode_id=episode.recovery_episode_id,
                expected_revision=episode.revision,
                completed_at=now,
                terminal_reason="human_decision_port_unavailable",
            )
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.UNAVAILABLE,
                episode=episode,
            )
        worker = self._load_worker(episode.worker_instance_id)
        if worker is not None:
            transition = self._transition_lifecycle(
                worker=worker,
                target_state=WorkerLifecycleState.WAITING_FOR_HUMAN,
                reason="recovery_human_decision",
            )
            if transition.disposition in {
                WorkerRecoveryLifecycleTransitionDisposition.CONFLICT,
                WorkerRecoveryLifecycleTransitionDisposition.NOT_FOUND,
                WorkerRecoveryLifecycleTransitionDisposition.INVALID,
            }:
                return WorkerRecoveryOrchestrationResult(
                    disposition=WorkerRecoveryOrchestrationDisposition.CONFLICT,
                    episode=episode,
                )
        episode = self._episode_repository.mark_waiting_for_human(
            recovery_episode_id=episode.recovery_episode_id,
            expected_revision=episode.revision,
            human_decision_ref=str(human_ref),
            updated_at=now,
        )
        return WorkerRecoveryOrchestrationResult(
            disposition=WorkerRecoveryOrchestrationDisposition.WAITING_FOR_HUMAN,
            episode=episode,
        )

    def _handle_wait(
        self,
        episode: WorkerRecoveryEpisode,
        *,
        decision,
        now: datetime,
    ) -> WorkerRecoveryOrchestrationResult:
        if episode.status is RecoveryEpisodeStatus.WAITING:
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.WAITING,
                episode=episode,
            )
        worker = self._load_worker(episode.worker_instance_id)
        if worker is not None:
            transition = self._transition_lifecycle(
                worker=worker,
                target_state=WorkerLifecycleState.WAITING_EXTERNAL,
                reason="recovery_dependency_wait",
            )
            if transition.disposition in {
                WorkerRecoveryLifecycleTransitionDisposition.CONFLICT,
                WorkerRecoveryLifecycleTransitionDisposition.NOT_FOUND,
                WorkerRecoveryLifecycleTransitionDisposition.INVALID,
            }:
                return WorkerRecoveryOrchestrationResult(
                    disposition=WorkerRecoveryOrchestrationDisposition.CONFLICT,
                    episode=episode,
                )
        episode = self._episode_repository.mark_waiting(
            recovery_episode_id=episode.recovery_episode_id,
            expected_revision=episode.revision,
            next_retry_at=decision.retry_after,
            dependency_ref=decision.dependency_ref,
            updated_at=now,
        )
        return WorkerRecoveryOrchestrationResult(
            disposition=WorkerRecoveryOrchestrationDisposition.WAITING,
            episode=episode,
        )

    def _handle_capability_deferred(
        self,
        episode: WorkerRecoveryEpisode,
        *,
        request: WorkerRecoveryOrchestrationRequest,
        now: datetime,
    ) -> WorkerRecoveryOrchestrationResult:
        capability = self._capability_port.request_acquisition(
            WorkerCapabilityAcquisitionRequest(
                episode=episode,
                capability_missing_ref=request.decision.resume_target_ref,
            )
        )
        if capability.disposition is PortAvailabilityDisposition.UNAVAILABLE:
            episode = self._episode_repository.mark_escalated(
                recovery_episode_id=episode.recovery_episode_id,
                expected_revision=episode.revision,
                completed_at=now,
                terminal_reason="capability_acquisition_unavailable",
            )
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.UNAVAILABLE,
                episode=episode,
            )
        episode = self._episode_repository.mark_escalated(
            recovery_episode_id=episode.recovery_episode_id,
            expected_revision=episode.revision,
            completed_at=now,
            terminal_reason="capability_acquisition_deferred",
        )
        return WorkerRecoveryOrchestrationResult(
            disposition=WorkerRecoveryOrchestrationDisposition.ESCALATED,
            episode=episode,
        )

    async def _handle_replan(
        self,
        episode: WorkerRecoveryEpisode,
        *,
        request: WorkerRecoveryOrchestrationRequest,
        dispatch_request: WorkerExecutionDispatchRequest[InputT, OutputT] | None,
        bounds: RecoveryExecutionBounds | None,
        now: datetime,
    ) -> WorkerRecoveryOrchestrationResult:
        alt_ref = request.decision.resume_target_ref or request.decision.source_ref
        replan = self._replan_port.prepare_alternative(
            WorkerRecoveryReplanRequest(episode=episode, alternative_path_ref=alt_ref),
        )
        if (
            replan.disposition is PortAvailabilityDisposition.UNAVAILABLE
            or replan.resume_target is None
        ):
            episode = self._episode_repository.mark_escalated(
                recovery_episode_id=episode.recovery_episode_id,
                expected_revision=episode.revision,
                completed_at=now,
                terminal_reason="replan_unavailable",
            )
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.UNAVAILABLE,
                episode=episode,
            )
        replanned_request = replace(request, resume_target=replan.resume_target)
        return await self._handle_retry(
            episode,
            request=replanned_request,
            dispatch_request=dispatch_request,
            bounds=bounds,
            now=now,
            strategy=RecoveryStrategy.REPLAN,
        )

    async def _handle_retry(
        self,
        episode: WorkerRecoveryEpisode,
        *,
        request: WorkerRecoveryOrchestrationRequest,
        dispatch_request: WorkerExecutionDispatchRequest[InputT, OutputT] | None,
        bounds: RecoveryExecutionBounds | None,
        now: datetime,
        strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
    ) -> WorkerRecoveryOrchestrationResult:
        max_attempts = episode.max_attempts or request.decision.max_attempts
        if max_attempts is None or max_attempts <= 0:
            episode = self._episode_repository.mark_escalated(
                recovery_episode_id=episode.recovery_episode_id,
                expected_revision=episode.revision,
                completed_at=now,
                terminal_reason="retry_unbounded",
            )
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.LIMIT_EXCEEDED,
                episode=episode,
            )
        if episode.attempt_count >= max_attempts:
            episode = self._episode_repository.mark_escalated(
                recovery_episode_id=episode.recovery_episode_id,
                expected_revision=episode.revision,
                completed_at=now,
                terminal_reason="attempt_limit_exceeded",
            )
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.LIMIT_EXCEEDED,
                episode=episode,
            )

        worker = self._load_worker(episode.worker_instance_id)
        if worker is None:
            episode = self._episode_repository.mark_failed(
                recovery_episode_id=episode.recovery_episode_id,
                expected_revision=episode.revision,
                completed_at=now,
                terminal_reason="worker_not_found",
            )
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.FAILED,
                episode=episode,
            )
        if worker.lifecycle_state in _INELIGIBLE_WORKER_LIFECYCLE_STATES:
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.FAILED,
                episode=episode,
            )

        stale = self._validate_source_freshness(request=request, now=now)
        if stale is not None:
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.STALE_SOURCE,
                episode=episode,
            )

        attempt_number = episode.attempt_count + 1
        claim = self._episode_repository.claim_attempt(
            recovery_episode_id=episode.recovery_episode_id,
            attempt_number=attempt_number,
            expected_revision=episode.revision,
            claimed_at=now,
        )
        if claim.status is WorkerRecoveryEpisodeClaimStatus.ALREADY_CLAIMED:
            if _episode_requires_reconciliation(claim.episode):
                return WorkerRecoveryOrchestrationResult(
                    disposition=WorkerRecoveryOrchestrationDisposition.RECONCILIATION_REQUIRED,
                    episode=claim.episode,
                )
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.ATTEMPT_DISPATCHED,
                episode=claim.episode,
            )
        if claim.status is WorkerRecoveryEpisodeClaimStatus.REVISION_CONFLICT:
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.CONFLICT,
                episode=claim.episode,
            )
        if claim.status is WorkerRecoveryEpisodeClaimStatus.TERMINAL:
            return _terminal_reentry_result(claim.episode)

        episode = claim.episode
        transition = self._transition_lifecycle(
            worker=worker,
            target_state=WorkerLifecycleState.RECOVERING,
            reason="recovery_attempt",
        )
        if transition.disposition in {
            WorkerRecoveryLifecycleTransitionDisposition.CONFLICT,
            WorkerRecoveryLifecycleTransitionDisposition.NOT_FOUND,
            WorkerRecoveryLifecycleTransitionDisposition.INVALID,
        }:
            episode = self._episode_repository.record_attempt_outcome(
                recovery_episode_id=episode.recovery_episode_id,
                expected_revision=episode.revision,
                attempt_number=attempt_number,
                finished_at=now,
                last_failure_ref="lifecycle_transition_conflict",
                next_retry_at=None,
                status=RecoveryEpisodeStatus.PENDING,
            )
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.CONFLICT,
                episode=episode,
            )
        if transition.worker is not None:
            worker = transition.worker

        if dispatch_request is None:
            episode = self._episode_repository.record_attempt_outcome(
                recovery_episode_id=episode.recovery_episode_id,
                expected_revision=episode.revision,
                attempt_number=attempt_number,
                finished_at=now,
                last_failure_ref="dispatch_request_missing",
                next_retry_at=None,
                status=RecoveryEpisodeStatus.PENDING,
            )
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.FAILED,
                episode=episode,
                attempt_result=_attempt_result(
                    episode=episode,
                    attempt_number=attempt_number,
                    strategy=strategy,
                    disposition=WorkerRecoveryAttemptDisposition.FAILED,
                    started_at=now,
                    finished_at=now,
                    failure_ref="dispatch_request_missing",
                ),
            )

        if dispatch_request.source.source_kind is not WorkerExecutionSourceKind.RECOVERY:
            episode = self._episode_repository.record_attempt_outcome(
                recovery_episode_id=episode.recovery_episode_id,
                expected_revision=episode.revision,
                attempt_number=attempt_number,
                finished_at=now,
                last_failure_ref="recovery_source_required",
                next_retry_at=None,
                status=RecoveryEpisodeStatus.PENDING,
            )
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.FAILED,
                episode=episode,
            )

        dispatch_result = await self._dispatch_port.dispatch_recovery(
            episode=episode,
            worker=worker,
            resume_target=request.resume_target,
            attempt_number=attempt_number,
            request=dispatch_request,
        )

        if dispatch_result.disposition is WorkerExecutionDispatchDisposition.DISPATCHED:
            execution_id = dispatch_result.correlation.execution_id
            if execution_id is not None:
                episode = self._episode_repository.record_execution(
                    recovery_episode_id=episode.recovery_episode_id,
                    attempt_number=attempt_number,
                    expected_revision=episode.revision,
                    execution_id=execution_id,
                    recorded_at=now,
                )
            episode, resume_result = self._resume_original_work(
                episode=episode,
                request=request,
                now=now,
                worker=worker,
            )
            if (
                resume_result.disposition
                is WorkerOriginalWorkResumeDisposition.RESUMED
            ):
                episode = self._episode_repository.mark_succeeded(
                    recovery_episode_id=episode.recovery_episode_id,
                    expected_revision=episode.revision,
                    completed_at=now,
                    terminal_reason="recovery_attempt_succeeded",
                )
                attempt = _attempt_result(
                    episode=episode,
                    attempt_number=attempt_number,
                    strategy=strategy,
                    disposition=WorkerRecoveryAttemptDisposition.SUCCEEDED,
                    started_at=now,
                    finished_at=now,
                    execution_id=execution_id,
                )
                return WorkerRecoveryOrchestrationResult(
                    disposition=WorkerRecoveryOrchestrationDisposition.RESUMED,
                    episode=episode,
                    attempt_result=attempt,
                    resume_intent=resume_result.resume_intent,
                )
            if (
                resume_result.disposition
                is WorkerOriginalWorkResumeDisposition.CONFLICT
            ):
                return WorkerRecoveryOrchestrationResult(
                    disposition=_orchestration_disposition_for_resume(
                        episode,
                        resume_result,
                    ),
                    episode=episode,
                    attempt_result=_attempt_result(
                        episode=episode,
                        attempt_number=attempt_number,
                        strategy=strategy,
                        disposition=WorkerRecoveryAttemptDisposition.SUCCEEDED,
                        started_at=now,
                        finished_at=now,
                        execution_id=execution_id,
                    ),
                )
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.UNAVAILABLE,
                episode=episode,
            )

        failure_ref = dispatch_result.disposition.value
        attempt_disposition = WorkerRecoveryAttemptDisposition.FAILED
        orchestration_disposition = WorkerRecoveryOrchestrationDisposition.FAILED
        if dispatch_result.disposition is WorkerExecutionDispatchDisposition.REJECTED:
            attempt_disposition = WorkerRecoveryAttemptDisposition.REJECTED
            orchestration_disposition = WorkerRecoveryOrchestrationDisposition.FAILED
            if (
                dispatch_result.rejection_reason
                is WorkerExecutionDispatchRejectionReason.BUDGET_DENIED
            ):
                orchestration_disposition = WorkerRecoveryOrchestrationDisposition.LIMIT_EXCEEDED
            if (
                dispatch_result.rejection_reason
                is WorkerExecutionDispatchRejectionReason.STALE_SOURCE
            ):
                orchestration_disposition = WorkerRecoveryOrchestrationDisposition.STALE_SOURCE
        elif dispatch_result.disposition is WorkerExecutionDispatchDisposition.UNAVAILABLE:
            attempt_disposition = WorkerRecoveryAttemptDisposition.UNAVAILABLE
            orchestration_disposition = WorkerRecoveryOrchestrationDisposition.UNAVAILABLE

        execution_id = dispatch_result.correlation.execution_id
        if execution_id is not None:
            episode = self._episode_repository.record_execution(
                recovery_episode_id=episode.recovery_episode_id,
                attempt_number=attempt_number,
                expected_revision=episode.revision,
                execution_id=execution_id,
                recorded_at=now,
            )

        next_status = RecoveryEpisodeStatus.PENDING
        if attempt_number >= max_attempts:
            episode = self._episode_repository.mark_failed(
                recovery_episode_id=episode.recovery_episode_id,
                expected_revision=episode.revision,
                completed_at=now,
                terminal_reason="attempt_failed_terminal",
                last_failure_ref=failure_ref,
            )
            orchestration_disposition = WorkerRecoveryOrchestrationDisposition.LIMIT_EXCEEDED
        else:
            episode = self._episode_repository.record_attempt_outcome(
                recovery_episode_id=episode.recovery_episode_id,
                expected_revision=episode.revision,
                attempt_number=attempt_number,
                finished_at=now,
                last_failure_ref=failure_ref,
                next_retry_at=request.decision.retry_after,
                status=next_status,
            )

        return WorkerRecoveryOrchestrationResult(
            disposition=orchestration_disposition,
            episode=episode,
            attempt_result=_attempt_result(
                episode=episode,
                attempt_number=attempt_number,
                strategy=strategy,
                disposition=attempt_disposition,
                started_at=now,
                finished_at=now,
                execution_id=execution_id,
                failure_ref=failure_ref,
                next_retry_at=request.decision.retry_after,
            ),
        )

    def _reconcile_in_progress_episode(
        self,
        episode: WorkerRecoveryEpisode,
        *,
        request: WorkerRecoveryOrchestrationRequest,
        now: datetime,
    ) -> WorkerRecoveryOrchestrationResult | None:
        if episode.status is not RecoveryEpisodeStatus.IN_PROGRESS:
            return None
        if _episode_requires_reconciliation(episode):
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.RECONCILIATION_REQUIRED,
                episode=episode,
            )
        if episode.continuity_resume_completed:
            return self._reconcile_pending_lifecycle_resume(
                episode,
                request=request,
                now=now,
            )
        if episode.last_execution_id is None:
            return None
        terminal = self._execution_outcome_reader.get_terminal_outcome(
            episode.last_execution_id,
        )
        if terminal.disposition is CanonicalExecutionTerminalDisposition.IN_PROGRESS:
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.ATTEMPT_DISPATCHED,
                episode=episode,
            )
        if terminal.disposition is CanonicalExecutionTerminalDisposition.UNAVAILABLE:
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.UNAVAILABLE,
                episode=episode,
            )
        if terminal.disposition is CanonicalExecutionTerminalDisposition.SUCCEEDED:
            worker = self._load_worker(episode.worker_instance_id)
            if worker is None:
                episode = self._episode_repository.mark_failed(
                    recovery_episode_id=episode.recovery_episode_id,
                    expected_revision=episode.revision,
                    completed_at=now,
                    terminal_reason="worker_not_found",
                )
                return WorkerRecoveryOrchestrationResult(
                    disposition=WorkerRecoveryOrchestrationDisposition.FAILED,
                    episode=episode,
                )
            episode, resume_result = self._resume_original_work(
                episode=episode,
                request=request,
                now=now,
                worker=worker,
            )
            if (
                resume_result.disposition
                is WorkerOriginalWorkResumeDisposition.RESUMED
            ):
                episode = self._episode_repository.mark_succeeded(
                    recovery_episode_id=episode.recovery_episode_id,
                    expected_revision=episode.revision,
                    completed_at=now,
                    terminal_reason="execution_terminal_success",
                )
                return WorkerRecoveryOrchestrationResult(
                    disposition=WorkerRecoveryOrchestrationDisposition.RESUMED,
                    episode=episode,
                    resume_intent=resume_result.resume_intent,
                )
            if (
                resume_result.disposition
                is WorkerOriginalWorkResumeDisposition.CONFLICT
            ):
                return WorkerRecoveryOrchestrationResult(
                    disposition=_orchestration_disposition_for_resume(
                        episode,
                        resume_result,
                    ),
                    episode=episode,
                )
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.UNAVAILABLE,
                episode=episode,
            )
        episode = self._episode_repository.mark_failed(
            recovery_episode_id=episode.recovery_episode_id,
            expected_revision=episode.revision,
            completed_at=now,
            terminal_reason="execution_terminal_failure",
            last_failure_ref=terminal.failure_ref,
        )
        return WorkerRecoveryOrchestrationResult(
            disposition=WorkerRecoveryOrchestrationDisposition.FAILED,
            episode=episode,
        )

    def _reconcile_pending_lifecycle_resume(
        self,
        episode: WorkerRecoveryEpisode,
        *,
        request: WorkerRecoveryOrchestrationRequest,
        now: datetime,
    ) -> WorkerRecoveryOrchestrationResult:
        worker = self._load_worker(episode.worker_instance_id)
        if worker is None:
            episode = self._episode_repository.mark_failed(
                recovery_episode_id=episode.recovery_episode_id,
                expected_revision=episode.revision,
                completed_at=now,
                terminal_reason="worker_not_found",
            )
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.FAILED,
                episode=episode,
            )
        episode, resume_result = self._resume_original_work(
            episode=episode,
            request=request,
            now=now,
            worker=worker,
        )
        if resume_result.disposition is WorkerOriginalWorkResumeDisposition.RESUMED:
            episode = self._episode_repository.mark_succeeded(
                recovery_episode_id=episode.recovery_episode_id,
                expected_revision=episode.revision,
                completed_at=now,
                terminal_reason="recovery_resume_completed",
            )
            return WorkerRecoveryOrchestrationResult(
                disposition=WorkerRecoveryOrchestrationDisposition.RESUMED,
                episode=episode,
                resume_intent=resume_result.resume_intent,
            )
        return WorkerRecoveryOrchestrationResult(
            disposition=_orchestration_disposition_for_resume(episode, resume_result),
            episode=episode,
        )

    def _resume_original_work(
        self,
        *,
        episode: WorkerRecoveryEpisode,
        request: WorkerRecoveryOrchestrationRequest,
        now: datetime,
        worker: WorkerInstance,
    ) -> tuple[WorkerRecoveryEpisode, WorkerOriginalWorkResumeResult]:
        episode, continuity_revision, continuity_conflict = self._apply_continuity_for_resume(
            episode=episode,
            request=request,
            now=now,
        )
        if continuity_conflict:
            return episode, WorkerOriginalWorkResumeResult(
                disposition=WorkerOriginalWorkResumeDisposition.CONFLICT,
                continuity_revision=continuity_revision,
            )

        resume_state = _resume_lifecycle_state(
            pre_recovery=episode.pre_recovery_lifecycle_state,
            worker=worker,
        )
        transition = self._transition_lifecycle(
            worker=worker,
            target_state=resume_state,
            reason="recovery_resume_original_work",
        )
        if transition.disposition in {
            WorkerRecoveryLifecycleTransitionDisposition.CONFLICT,
            WorkerRecoveryLifecycleTransitionDisposition.NOT_FOUND,
            WorkerRecoveryLifecycleTransitionDisposition.INVALID,
        }:
            return episode, WorkerOriginalWorkResumeResult(
                disposition=WorkerOriginalWorkResumeDisposition.CONFLICT,
                continuity_revision=continuity_revision,
            )
        resume_intent = WorkerOriginalWorkResumeIntent(
            worker_instance_id=episode.worker_instance_id,
            recovery_episode_id=episode.recovery_episode_id,
            original_source=request.original_source,
            resume_target=request.resume_target,
            continuity_revision=continuity_revision,
            created_at=now,
        )
        return episode, WorkerOriginalWorkResumeResult(
            disposition=WorkerOriginalWorkResumeDisposition.RESUMED,
            resume_intent=resume_intent,
            continuity_revision=continuity_revision,
        )

    def _apply_continuity_for_resume(
        self,
        *,
        episode: WorkerRecoveryEpisode,
        request: WorkerRecoveryOrchestrationRequest,
        now: datetime,
    ) -> tuple[WorkerRecoveryEpisode, Revision, bool]:
        continuity = self._continuity_repository.get(
            worker_instance_id=episode.worker_instance_id,
        )
        continuity_revision = (
            continuity.revision if continuity is not None else initial_revision()
        )
        if episode.continuity_resume_completed:
            assert episode.continuity_resume_revision is not None
            return episode, episode.continuity_resume_revision, False
        if continuity is None or request.continuity_expected_revision is None:
            return episode, continuity_revision, False

        owned_refs = frozenset(request.evidence_refs)
        refs_to_clear = tuple(
            ref for ref in owned_refs if ref in continuity.unresolved_problem_refs
        )
        if not refs_to_clear:
            episode = self._episode_repository.record_continuity_resume(
                recovery_episode_id=episode.recovery_episode_id,
                expected_revision=episode.revision,
                continuity_resume_revision=continuity.revision,
                recorded_at=now,
            )
            return episode, continuity.revision, False

        try:
            updated = replace(
                continuity,
                unresolved_problem_refs=tuple(
                    ref
                    for ref in continuity.unresolved_problem_refs
                    if ref not in owned_refs
                ),
                revision=continuity.revision,
            )
            persisted = self._continuity_repository.replace(
                updated,
                expected_revision=request.continuity_expected_revision,
            )
            continuity_revision = persisted.revision
        except AutonomousWorkRevisionConflict:
            return episode, continuity_revision, True

        episode = self._episode_repository.record_continuity_resume(
            recovery_episode_id=episode.recovery_episode_id,
            expected_revision=episode.revision,
            continuity_resume_revision=continuity_revision,
            recorded_at=now,
        )
        return episode, continuity_revision, False

    def _validate_source_freshness(
        self,
        *,
        request: WorkerRecoveryOrchestrationRequest,
        now: datetime,
    ) -> WorkerRecoveryOrchestrationDisposition | None:
        target = request.resume_target
        if target.goal_id is None or target.goal_revision is None:
            return None
        goal = self._worker_goal_repository.get(goal_id=target.goal_id)
        if goal is None:
            return WorkerRecoveryOrchestrationDisposition.STALE_SOURCE
        if goal.revision != target.goal_revision:
            return WorkerRecoveryOrchestrationDisposition.STALE_SOURCE
        if goal.status is not WorkerGoalStatus.ACTIVE:
            return WorkerRecoveryOrchestrationDisposition.STALE_SOURCE
        return None

    def _load_worker(self, worker_instance_id: WorkerInstanceId) -> WorkerInstance | None:
        return self._worker_instance_repository.get(worker_instance_id=worker_instance_id)

    def _transition_lifecycle(
        self,
        *,
        worker: WorkerInstance,
        target_state: WorkerLifecycleState,
        reason: str,
    ) -> WorkerRecoveryLifecycleTransitionOutcome:
        if worker.lifecycle_state == target_state:
            return WorkerRecoveryLifecycleTransitionOutcome(
                disposition=WorkerRecoveryLifecycleTransitionDisposition.UNCHANGED,
                worker=worker,
            )
        try:
            result = self._lifecycle_service.transition(
                WorkerLifecycleTransitionRequest(
                    worker_instance_id=worker.worker_instance_id,
                    expected_revision=worker.revision,
                    expected_state=worker.lifecycle_state,
                    target_state=target_state,
                    transition_reason=reason,
                )
            )
        except AutonomousWorkEntityNotFound:
            return WorkerRecoveryLifecycleTransitionOutcome(
                disposition=WorkerRecoveryLifecycleTransitionDisposition.NOT_FOUND,
            )
        except AutonomousWorkRevisionConflict:
            return WorkerRecoveryLifecycleTransitionOutcome(
                disposition=WorkerRecoveryLifecycleTransitionDisposition.CONFLICT,
            )
        except AutonomousWorkLifecycleStateConflict:
            return WorkerRecoveryLifecycleTransitionOutcome(
                disposition=WorkerRecoveryLifecycleTransitionDisposition.CONFLICT,
            )
        except AutonomousWorkInvalidLifecycleTransition:
            return WorkerRecoveryLifecycleTransitionOutcome(
                disposition=WorkerRecoveryLifecycleTransitionDisposition.INVALID,
            )
        disposition = (
            WorkerRecoveryLifecycleTransitionDisposition.APPLIED
            if result.changed
            else WorkerRecoveryLifecycleTransitionDisposition.UNCHANGED
        )
        return WorkerRecoveryLifecycleTransitionOutcome(
            disposition=disposition,
            worker=result.worker_instance,
        )


def _orchestration_disposition_for_resume(
    episode: WorkerRecoveryEpisode,
    resume_result: WorkerOriginalWorkResumeResult,
) -> WorkerRecoveryOrchestrationDisposition:
    if resume_result.disposition is not WorkerOriginalWorkResumeDisposition.CONFLICT:
        return WorkerRecoveryOrchestrationDisposition.UNAVAILABLE
    if episode.continuity_resume_completed:
        return WorkerRecoveryOrchestrationDisposition.CONFLICT
    return WorkerRecoveryOrchestrationDisposition.STALE_CONTINUITY


def _episode_requires_reconciliation(episode: WorkerRecoveryEpisode) -> bool:
    """Return whether a claimed attempt lacks canonical execution binding."""
    return (
        episode.status is RecoveryEpisodeStatus.IN_PROGRESS
        and episode.claimed_attempt_number is not None
        and episode.last_execution_id is None
    )


def _episode_from_request(
    request: WorkerRecoveryOrchestrationRequest,
    *,
    started_at: datetime,
) -> WorkerRecoveryEpisode:
    decision = request.decision
    episode_id = derive_recovery_episode_id(
        worker_instance_id=request.original_source.worker_instance_id,
        obstacle_id=decision.obstacle_id,
        recovery_decision_id=decision.decision_id,
    )
    human_ref = (
        str(decision.human_decision_ref) if decision.human_decision_ref is not None else None
    )
    return WorkerRecoveryEpisode(
        recovery_episode_id=episode_id,
        worker_instance_id=request.original_source.worker_instance_id,
        obstacle_id=decision.obstacle_id,
        recovery_decision_id=decision.decision_id,
        decision_policy_version=decision.decision_policy_version,
        strategy=decision.strategy,
        original_source=request.original_source,
        resume_target=request.resume_target,
        started_at=started_at,
        status=RecoveryEpisodeStatus.PENDING,
        attempt_count=0,
        revision=initial_revision(),
        max_attempts=decision.max_attempts,
        pre_recovery_lifecycle_state=request.pre_recovery_lifecycle_state,
        dependency_ref=decision.dependency_ref,
        human_decision_ref=human_ref,
    )


def _deadline_exceeded(
    episode: WorkerRecoveryEpisode,
    bounds: RecoveryExecutionBounds | None,
    now: datetime,
) -> bool:
    if bounds is None:
        return False
    if bounds.deadline is not None and now > bounds.deadline:
        return True
    if bounds.max_elapsed_seconds is not None:
        elapsed = (now - episode.started_at).total_seconds()
        return elapsed > bounds.max_elapsed_seconds
    return False


def _terminal_reentry_result(
    episode: WorkerRecoveryEpisode,
) -> WorkerRecoveryOrchestrationResult:
    if episode.status is RecoveryEpisodeStatus.SUCCEEDED:
        disposition = WorkerRecoveryOrchestrationDisposition.ALREADY_SUCCEEDED
    else:
        disposition = WorkerRecoveryOrchestrationDisposition.ALREADY_TERMINAL
    return WorkerRecoveryOrchestrationResult(
        disposition=disposition,
        episode=episode,
    )


def _resume_lifecycle_state(
    *,
    pre_recovery: WorkerLifecycleState | None,
    worker: WorkerInstance,
) -> WorkerLifecycleState:
    if pre_recovery is not None and pre_recovery not in _INELIGIBLE_WORKER_LIFECYCLE_STATES:
        return pre_recovery
    if worker.lifecycle_state is WorkerLifecycleState.IDLE:
        return WorkerLifecycleState.IDLE
    return WorkerLifecycleState.WORKING


def _attempt_result(
    *,
    episode: WorkerRecoveryEpisode,
    attempt_number: int,
    strategy: RecoveryStrategy,
    disposition: WorkerRecoveryAttemptDisposition,
    started_at: datetime,
    finished_at: datetime,
    execution_id=None,
    failure_ref: str | None = None,
    next_retry_at: datetime | None = None,
) -> WorkerRecoveryAttemptResult:
    return WorkerRecoveryAttemptResult(
        recovery_episode_id=episode.recovery_episode_id,
        attempt_number=attempt_number,
        attempt_id=derive_recovery_attempt_id(
            recovery_episode_id=episode.recovery_episode_id,
            attempt_number=attempt_number,
        ),
        strategy=strategy,
        disposition=disposition,
        started_at=started_at,
        finished_at=finished_at,
        execution_id=execution_id,
        failure_ref=failure_ref,
        next_retry_at=next_retry_at,
    )

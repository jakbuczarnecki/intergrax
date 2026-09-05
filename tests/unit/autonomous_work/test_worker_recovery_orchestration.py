# © Artur Czarnecki. All rights reserved.

"""AW-6B — worker recovery orchestration tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

from unittest.mock import patch

import pytest

from intergrax.autonomous_work.in_memory_recovery_episode_repository import (
    InMemoryWorkerRecoveryEpisodeRepository,
)
from intergrax.autonomous_work.in_memory_repository import (
    InMemoryResponsibilityRepository,
    InMemoryWorkContinuityStateRepository,
    InMemoryWorkerGoalRepository,
    InMemoryWorkerInstanceRepository,
)
from intergrax.autonomous_work.lifecycle import WorkerLifecycleService
from intergrax.autonomous_work.recovery_orchestration_ports import (
    CanonicalExecutionTerminalDisposition,
    CanonicalExecutionTerminalOutcome,
    HumanDecisionRequest,
    HumanDecisionRequestResult,
    PortAvailabilityDisposition,
)
from intergrax.autonomous_work.repository import AutonomousWorkRevisionConflict
from intergrax.autonomous_work.worker_recovery_decision_service import (
    WorkerRecoveryDecisionService,
)
from intergrax.autonomous_work.worker_recovery_orchestration_service import (
    WorkerRecoveryOrchestrationService,
)
from intergrax.contracts.autonomous_work.execution_dispatch import (
    WorkerExecutionCorrelation,
    WorkerExecutionDispatchDisposition,
    WorkerExecutionDispatchRejectionReason,
    WorkerExecutionDispatchResult,
    WorkerExecutionSource,
    WorkerExecutionSourceKind,
)
from intergrax.contracts.autonomous_work.lifecycle import WorkerLifecycleState
from intergrax.contracts.autonomous_work.obstacle_recovery import (
    RecoveryDecisionReasonCode,
    RecoveryStrategy,
    WorkerObstacleEvidence,
    WorkerObstacleKind,
    WorkerObstacleSourceKind,
    WorkerRecoveryDecision,
)
from intergrax.contracts.autonomous_work.recovery_orchestration import (
    RecoveryEpisodeStatus,
    RecoveryExecutionBounds,
    WorkerOriginalWorkSource,
    WorkerRecoveryOrchestrationDisposition,
    WorkerRecoveryOrchestrationRequest,
    WorkerRecoveryResumeTarget,
    WorkerRecoveryResumeTargetKind,
    derive_recovery_episode_id,
)
from intergrax.contracts.autonomous_work.references import (
    ExternalDependencyReference,
    HumanPendingReference,
    ProblemReference,
)
from intergrax.contracts.autonomous_work.revision import Revision
from intergrax.contracts.execution_identity import ExecutionId, mint_attempt_id, mint_execution_id, mint_run_id
from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
from intergrax.contracts.autonomous_work.ids import mint_wake_up_id
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = pytest.mark.unit

_UTC = UTC
_NOW = datetime(2026, 9, 4, 14, 0, tzinfo=_UTC)
_WORKER_ID = contract_suite.mint_worker_instance_id()
_GOAL_ID = contract_suite.mint_worker_goal_id()
_RESP_ID = contract_suite.mint_responsibility_id()
_WAKE_UP_ID = mint_wake_up_id()
_EVIDENCE_REF = ProblemReference("problem/evidence/recovery-1")


class RecordingHumanDecisionPort:
    def __init__(self) -> None:
        self.calls: list[HumanDecisionRequest] = []

    def request_human_decision(
        self,
        request: HumanDecisionRequest,
    ) -> HumanDecisionRequestResult:
        self.calls.append(request)
        return HumanDecisionRequestResult(
            disposition=PortAvailabilityDisposition.AVAILABLE,
            correlation_ref="hitl/correlation/1",
        )


class RecordingRecoveryDispatchPort:
    def __init__(
        self,
        *,
        disposition: WorkerExecutionDispatchDisposition = (
            WorkerExecutionDispatchDisposition.DISPATCHED
        ),
        rejection_reason: WorkerExecutionDispatchRejectionReason | None = None,
    ) -> None:
        self.calls: list[object] = []
        self._disposition = disposition
        self._rejection_reason = rejection_reason

    async def dispatch_recovery(self, **kwargs) -> WorkerExecutionDispatchResult[object]:
        self.calls.append(kwargs)
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        if self._disposition in {
            WorkerExecutionDispatchDisposition.REJECTED,
            WorkerExecutionDispatchDisposition.UNAVAILABLE,
        }:
            return WorkerExecutionDispatchResult(
                disposition=self._disposition,
                correlation=WorkerExecutionCorrelation(
                    worker_instance_id=_WORKER_ID,
                    source=WorkerExecutionSource(
                        source_kind=WorkerExecutionSourceKind.RECOVERY,
                        source_ref="recovery/episode/1",
                    ),
                    run_id=None,
                    attempt_id=None,
                    execution_id=None,
                ),
                rejection_reason=self._rejection_reason,
            )
        return WorkerExecutionDispatchResult(
            disposition=self._disposition,
            correlation=WorkerExecutionCorrelation(
                worker_instance_id=_WORKER_ID,
                source=WorkerExecutionSource(
                    source_kind=WorkerExecutionSourceKind.RECOVERY,
                    source_ref="recovery/episode/1",
                ),
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            ),
            rejection_reason=self._rejection_reason,
            failure_reason=(
                "dispatch_failed"
                if self._disposition is WorkerExecutionDispatchDisposition.FAILED
                else None
            ),
        )


@dataclass(frozen=True)
class ProbePayload:
    value: str


def _clock() -> datetime:
    return _NOW


def _original_source() -> WorkerOriginalWorkSource:
    return WorkerOriginalWorkSource(
        worker_instance_id=_WORKER_ID,
        source_kind=WorkerObstacleSourceKind.EXECUTION_FAILURE,
        source_ref="execution/terminal/failed-1",
    )


def _resume_target(**overrides) -> WorkerRecoveryResumeTarget:
    base = WorkerRecoveryResumeTarget(
        kind=WorkerRecoveryResumeTargetKind.GOAL_DECISION,
        source_ref=str(_GOAL_ID),
        goal_id=_GOAL_ID,
        goal_revision=Revision(0),
        responsibility_id=_RESP_ID,
        wake_up_id=_WAKE_UP_ID,
        requested_scopes=("workspace.read",),
    )
    if not overrides:
        return base
    return replace(base, **overrides)


def _decision(**overrides) -> WorkerRecoveryDecision:
    obstacle_id = f"{_WORKER_ID}:execution_failure:execution/terminal/failed-1:occ-1"
    base = WorkerRecoveryDecision(
        decision_id=f"{obstacle_id}:aw-6a.v1",
        obstacle_id=obstacle_id,
        obstacle_kind=WorkerObstacleKind.TRANSIENT_FAILURE,
        strategy=RecoveryStrategy.RETRY,
        decision_reason_code=RecoveryDecisionReasonCode.TRANSIENT_RETRY_BOUNDED,
        evidence_refs=(_EVIDENCE_REF,),
        decided_at=_NOW,
        source_ref="execution/terminal/failed-1",
        max_attempts=2,
        resume_target_ref=str(_GOAL_ID),
    )
    if not overrides:
        return base
    return replace(base, **overrides)


def _orchestration_request(**overrides) -> WorkerRecoveryOrchestrationRequest:
    base = WorkerRecoveryOrchestrationRequest(
        decision=_decision(),
        original_source=_original_source(),
        resume_target=_resume_target(),
        pre_recovery_lifecycle_state=WorkerLifecycleState.WORKING,
        evidence_refs=(_EVIDENCE_REF,),
    )
    if not overrides:
        return base
    return replace(base, **overrides)


def _dispatch_request():
    from intergrax.contracts.autonomous_work.execution_dispatch import (
        WorkerExecutionDispatchRequest,
    )

    return WorkerExecutionDispatchRequest(
        worker_instance_id=_WORKER_ID,
        worker_revision=Revision(0),
        requested_scopes=("workspace.read",),
        runtime_request=ExecutionRequest(
            input=ProbePayload(value="retry"),
            capabilities=frozenset({ExecutionCapability.AGENT}),
        ),
        source=WorkerExecutionSource(
            source_kind=WorkerExecutionSourceKind.RECOVERY,
            source_ref="recovery/episode/1",
        ),
        requested_at=_NOW,
        goal_id=_GOAL_ID,
        goal_revision=Revision(0),
        responsibility_id=_RESP_ID,
        wake_up_id=_WAKE_UP_ID,
    )


def _harness(
    *,
    dispatch_port: RecordingRecoveryDispatchPort | None = None,
    human_port: RecordingHumanDecisionPort | None = None,
) -> tuple[WorkerRecoveryOrchestrationService, dict[str, object]]:
    worker_repo = InMemoryWorkerInstanceRepository()
    goal_repo = InMemoryWorkerGoalRepository()
    responsibility_repo = InMemoryResponsibilityRepository()
    continuity_repo = InMemoryWorkContinuityStateRepository()
    episode_repo = InMemoryWorkerRecoveryEpisodeRepository()
    definition = contract_suite.worker_definition()
    from intergrax.autonomous_work.in_memory_repository import InMemoryWorkerDefinitionRepository

    definition_repo = InMemoryWorkerDefinitionRepository()
    definition_repo.create(definition)
    worker = contract_suite.worker_instance(
        worker_instance_id=_WORKER_ID,
        worker_definition_id=definition.worker_definition_id,
        lifecycle_state=WorkerLifecycleState.WORKING,
    )
    worker_repo.create(worker)
    responsibility = contract_suite.responsibility(
        responsibility_id=_RESP_ID,
        worker_instance_id=_WORKER_ID,
    )
    responsibility_repo.create(responsibility)
    goal = contract_suite.worker_goal(
        goal_id=_GOAL_ID,
        responsibility_id=_RESP_ID,
    )
    goal_repo.create(goal)
    continuity_repo.create(contract_suite.continuity_state(worker_instance_ref=_WORKER_ID))
    dispatch = dispatch_port or RecordingRecoveryDispatchPort()
    human = human_port or RecordingHumanDecisionPort()
    service = WorkerRecoveryOrchestrationService(
        episode_repository=episode_repo,
        worker_instance_repository=worker_repo,
        worker_goal_repository=goal_repo,
        continuity_repository=continuity_repo,
        lifecycle_service=WorkerLifecycleService(repository=worker_repo, clock=_clock),
        dispatch_port=dispatch,
        human_decision_port=human,
        clock=_clock,
    )
    return service, {
        "dispatch": dispatch,
        "human": human,
        "episode_repo": episode_repo,
        "worker_repo": worker_repo,
        "goal_repo": goal_repo,
        "continuity_repo": continuity_repo,
    }


@pytest.mark.asyncio
async def test_stop_strategy_zero_dispatch() -> None:
    service, ctx = _harness()
    request = _orchestration_request(
        decision=_decision(
            strategy=RecoveryStrategy.STOP,
            decision_reason_code=RecoveryDecisionReasonCode.POLICY_DENY_STOP,
            max_attempts=None,
            obstacle_kind=WorkerObstacleKind.POLICY_DENIED,
        ),
    )
    result = await service.orchestrate(request)
    assert result.disposition is WorkerRecoveryOrchestrationDisposition.STOPPED
    assert result.episode.status is RecoveryEpisodeStatus.STOPPED
    assert ctx["dispatch"].calls == []


@pytest.mark.asyncio
async def test_escalate_strategy_zero_dispatch() -> None:
    service, ctx = _harness()
    request = _orchestration_request(
        decision=_decision(
            strategy=RecoveryStrategy.ESCALATE,
            decision_reason_code=RecoveryDecisionReasonCode.CREDENTIAL_ESCALATE,
            max_attempts=None,
            obstacle_kind=WorkerObstacleKind.CREDENTIAL_UNAVAILABLE,
        ),
    )
    result = await service.orchestrate(request)
    assert result.disposition is WorkerRecoveryOrchestrationDisposition.ESCALATED
    assert ctx["dispatch"].calls == []


@pytest.mark.asyncio
async def test_unknown_escalate_zero_dispatch() -> None:
    service, ctx = _harness()
    request = _orchestration_request(
        decision=_decision(
            strategy=RecoveryStrategy.ESCALATE,
            decision_reason_code=RecoveryDecisionReasonCode.UNKNOWN_ESCALATE,
            max_attempts=None,
            obstacle_kind=WorkerObstacleKind.UNKNOWN,
        ),
    )
    result = await service.orchestrate(request)
    assert result.disposition is WorkerRecoveryOrchestrationDisposition.ESCALATED
    assert ctx["dispatch"].calls == []


@pytest.mark.asyncio
async def test_quarantine_uses_lifecycle_service() -> None:
    service, ctx = _harness()
    request = _orchestration_request(
        decision=_decision(
            strategy=RecoveryStrategy.QUARANTINE,
            decision_reason_code=RecoveryDecisionReasonCode.SUSPICIOUS_QUARANTINE,
            max_attempts=None,
            obstacle_kind=WorkerObstacleKind.SUSPICIOUS_OR_UNSAFE,
        ),
    )
    result = await service.orchestrate(request)
    assert result.disposition is WorkerRecoveryOrchestrationDisposition.QUARANTINED
    worker = ctx["worker_repo"].get(worker_instance_id=_WORKER_ID)
    assert worker is not None
    assert worker.lifecycle_state is WorkerLifecycleState.QUARANTINED
    assert ctx["dispatch"].calls == []


@pytest.mark.asyncio
async def test_human_wait_invokes_hitl_once() -> None:
    human = RecordingHumanDecisionPort()
    service, ctx = _harness(human_port=human)
    request = _orchestration_request(
        decision=_decision(
            strategy=RecoveryStrategy.REQUEST_HUMAN_DECISION,
            decision_reason_code=RecoveryDecisionReasonCode.HUMAN_DECISION_REQUIRED,
            max_attempts=None,
            obstacle_kind=WorkerObstacleKind.HUMAN_DECISION_REQUIRED,
            human_decision_ref=HumanPendingReference("human/pending/1"),
        ),
    )
    first = await service.orchestrate(request)
    second = await service.orchestrate(request)
    assert first.disposition is WorkerRecoveryOrchestrationDisposition.WAITING_FOR_HUMAN
    assert second.disposition is WorkerRecoveryOrchestrationDisposition.WAITING_FOR_HUMAN
    assert len(human.calls) == 1
    assert ctx["dispatch"].calls == []


@pytest.mark.asyncio
async def test_dependency_wait_durable_no_dispatch() -> None:
    service, ctx = _harness()
    request = _orchestration_request(
        decision=_decision(
            strategy=RecoveryStrategy.WAIT,
            decision_reason_code=RecoveryDecisionReasonCode.DEPENDENCY_WAIT,
            max_attempts=None,
            obstacle_kind=WorkerObstacleKind.DEPENDENCY_UNAVAILABLE,
            dependency_ref=ExternalDependencyReference("external/vendor-api"),
        ),
    )
    result = await service.orchestrate(request)
    assert result.disposition is WorkerRecoveryOrchestrationDisposition.WAITING
    assert result.episode.status is RecoveryEpisodeStatus.WAITING
    assert ctx["dispatch"].calls == []


@pytest.mark.asyncio
async def test_throttle_preserves_retry_time() -> None:
    service, _ = _harness()
    retry_after = _NOW + timedelta(minutes=5)
    request = _orchestration_request(
        decision=_decision(
            strategy=RecoveryStrategy.THROTTLE,
            decision_reason_code=RecoveryDecisionReasonCode.RATE_LIMIT_THROTTLE,
            max_attempts=None,
            obstacle_kind=WorkerObstacleKind.RATE_LIMITED,
            retry_after=retry_after,
        ),
    )
    result = await service.orchestrate(request)
    assert result.disposition is WorkerRecoveryOrchestrationDisposition.WAITING
    assert result.episode.next_retry_at == retry_after


@pytest.mark.asyncio
async def test_retry_success_resumes_original_work() -> None:
    service, ctx = _harness()
    request = _orchestration_request()
    result = await service.orchestrate(request, dispatch_request=_dispatch_request())
    assert result.disposition is WorkerRecoveryOrchestrationDisposition.RESUMED
    assert result.resume_intent is not None
    assert result.resume_intent.original_source == _original_source()
    assert result.resume_intent.resume_target == _resume_target()
    assert result.episode.status is RecoveryEpisodeStatus.SUCCEEDED
    assert len(ctx["dispatch"].calls) == 1


@pytest.mark.asyncio
async def test_retry_failure_then_success_bounded() -> None:
    failing = RecordingRecoveryDispatchPort(
        disposition=WorkerExecutionDispatchDisposition.FAILED,
    )
    service, ctx = _harness(dispatch_port=failing)
    request = _orchestration_request()
    first = await service.orchestrate(request, dispatch_request=_dispatch_request())
    assert first.episode.attempt_count == 1
    succeeding = RecordingRecoveryDispatchPort()
    service._dispatch_port = succeeding
    second = await service.orchestrate(request, dispatch_request=_dispatch_request())
    assert second.disposition is WorkerRecoveryOrchestrationDisposition.RESUMED
    assert second.episode.attempt_count == 2
    assert len(failing.calls) == 1
    assert len(succeeding.calls) == 1


@pytest.mark.asyncio
async def test_attempt_limit_exceeded_zero_dispatch() -> None:
    failing = RecordingRecoveryDispatchPort(
        disposition=WorkerExecutionDispatchDisposition.FAILED,
    )
    service, ctx = _harness(dispatch_port=failing)
    request = _orchestration_request(decision=_decision(max_attempts=2))
    await service.orchestrate(request, dispatch_request=_dispatch_request())
    second = await service.orchestrate(request, dispatch_request=_dispatch_request())
    assert second.episode.status is RecoveryEpisodeStatus.FAILED
    third = await service.orchestrate(request, dispatch_request=_dispatch_request())
    assert third.disposition is WorkerRecoveryOrchestrationDisposition.ALREADY_TERMINAL
    assert len(ctx["dispatch"].calls) == 2


@pytest.mark.asyncio
async def test_deadline_exceeded_zero_dispatch() -> None:
    service, ctx = _harness()
    request = _orchestration_request()
    bounds = RecoveryExecutionBounds(deadline=_NOW - timedelta(seconds=1))
    result = await service.orchestrate(
        request,
        dispatch_request=_dispatch_request(),
        bounds=bounds,
    )
    assert result.disposition is WorkerRecoveryOrchestrationDisposition.LIMIT_EXCEEDED
    assert ctx["dispatch"].calls == []


@pytest.mark.asyncio
async def test_budget_denied_limit_exceeded() -> None:
    dispatch = RecordingRecoveryDispatchPort(
        disposition=WorkerExecutionDispatchDisposition.REJECTED,
        rejection_reason=WorkerExecutionDispatchRejectionReason.BUDGET_DENIED,
    )
    service, ctx = _harness(dispatch_port=dispatch)
    result = await service.orchestrate(
        _orchestration_request(),
        dispatch_request=_dispatch_request(),
    )
    assert result.disposition is WorkerRecoveryOrchestrationDisposition.LIMIT_EXCEEDED
    assert len(ctx["dispatch"].calls) == 1


@pytest.mark.asyncio
async def test_paused_worker_zero_dispatch() -> None:
    service, ctx = _harness()
    worker = ctx["worker_repo"].get(worker_instance_id=_WORKER_ID)
    assert worker is not None
    paused = replace(worker, lifecycle_state=WorkerLifecycleState.PAUSED)
    ctx["worker_repo"].replace(paused, expected_revision=worker.revision)
    result = await service.orchestrate(
        _orchestration_request(),
        dispatch_request=_dispatch_request(),
    )
    assert result.disposition is WorkerRecoveryOrchestrationDisposition.FAILED
    assert ctx["dispatch"].calls == []


@pytest.mark.asyncio
async def test_stale_goal_zero_dispatch() -> None:
    service, ctx = _harness()
    goal = ctx["goal_repo"].get(goal_id=_GOAL_ID)
    assert goal is not None
    ctx["goal_repo"].replace(goal, expected_revision=goal.revision)
    bumped = ctx["goal_repo"].get(goal_id=_GOAL_ID)
    assert bumped is not None
    ctx["goal_repo"].replace(
        replace(bumped, objective="changed objective"),
        expected_revision=bumped.revision,
    )
    result = await service.orchestrate(
        _orchestration_request(),
        dispatch_request=_dispatch_request(),
    )
    assert result.disposition is WorkerRecoveryOrchestrationDisposition.STALE_SOURCE
    assert ctx["dispatch"].calls == []


@pytest.mark.asyncio
async def test_terminal_reentry_idempotent() -> None:
    service, _ = _harness()
    request = _orchestration_request()
    first = await service.orchestrate(request, dispatch_request=_dispatch_request())
    second = await service.orchestrate(request, dispatch_request=_dispatch_request())
    assert first.disposition is WorkerRecoveryOrchestrationDisposition.RESUMED
    assert second.disposition is WorkerRecoveryOrchestrationDisposition.ALREADY_SUCCEEDED


@pytest.mark.asyncio
async def test_duplicate_create_same_episode() -> None:
    service, ctx = _harness()
    request = _orchestration_request(
        decision=_decision(
            strategy=RecoveryStrategy.STOP,
            decision_reason_code=RecoveryDecisionReasonCode.POLICY_DENY_STOP,
            max_attempts=None,
            obstacle_kind=WorkerObstacleKind.POLICY_DENIED,
        ),
    )
    first = await service.orchestrate(request)
    second = await service.orchestrate(request)
    assert first.episode.recovery_episode_id == second.episode.recovery_episode_id


def test_aw6a_decision_service_still_usable() -> None:
    evidence = WorkerObstacleEvidence(
        worker_instance_id=_WORKER_ID,
        source_kind=WorkerObstacleSourceKind.POLICY_DECISION,
        source_ref="policy/deny/1",
        occurrence_identity="policy-1",
        observed_at=_NOW,
        policy_decision=None,
    )
    result = WorkerRecoveryDecisionService().decide(evidence, decided_at=_NOW)
    assert result.decision is not None


class StubExecutionOutcomeReader:
    def __init__(self, disposition: CanonicalExecutionTerminalDisposition) -> None:
        self._disposition = disposition

    def get_terminal_outcome(self, execution_id: ExecutionId) -> CanonicalExecutionTerminalOutcome:
        return CanonicalExecutionTerminalOutcome(
            disposition=self._disposition,
            execution_id=execution_id,
        )


@pytest.mark.asyncio
async def test_continuity_cas_conflict_blocks_resume() -> None:
    service, ctx = _harness()
    continuity = ctx["continuity_repo"].get(worker_instance_id=_WORKER_ID)
    assert continuity is not None
    request = _orchestration_request(continuity_expected_revision=continuity.revision)
    ctx["continuity_repo"].replace(
        replace(continuity, next_action_hint="stale-bump"),
        expected_revision=continuity.revision,
    )
    result = await service.orchestrate(request, dispatch_request=_dispatch_request())
    assert result.disposition is WorkerRecoveryOrchestrationDisposition.STALE_CONTINUITY
    assert result.resume_intent is None
    assert result.episode.status is RecoveryEpisodeStatus.IN_PROGRESS
    assert result.episode.last_execution_id is not None


@pytest.mark.asyncio
async def test_orphaned_claim_requires_reconciliation() -> None:
    service, ctx = _harness()
    request = _orchestration_request()
    episode_id = derive_recovery_episode_id(
        worker_instance_id=_WORKER_ID,
        obstacle_id=request.decision.obstacle_id,
        recovery_decision_id=request.decision.decision_id,
    )
    episode_repo = ctx["episode_repo"]
    from intergrax.contracts.autonomous_work.recovery_orchestration import (
        WorkerRecoveryEpisode,
    )
    from intergrax.contracts.autonomous_work.obstacle_recovery import DECISION_POLICY_VERSION
    from intergrax.contracts.autonomous_work.revision import initial_revision

    seed = WorkerRecoveryEpisode(
        recovery_episode_id=episode_id,
        worker_instance_id=_WORKER_ID,
        obstacle_id=request.decision.obstacle_id,
        recovery_decision_id=request.decision.decision_id,
        decision_policy_version=DECISION_POLICY_VERSION,
        strategy=request.decision.strategy,
        original_source=request.original_source,
        resume_target=request.resume_target,
        started_at=_NOW,
        status=RecoveryEpisodeStatus.PENDING,
        attempt_count=0,
        revision=initial_revision(),
        max_attempts=2,
        pre_recovery_lifecycle_state=WorkerLifecycleState.WORKING,
    )
    created = episode_repo.create_or_get(seed)
    claim = episode_repo.claim_attempt(
        recovery_episode_id=episode_id,
        attempt_number=1,
        expected_revision=created.episode.revision,
        claimed_at=_NOW,
    )
    assert claim.episode.claimed_attempt_number == 1
    result = await service.orchestrate(request, dispatch_request=_dispatch_request())
    assert result.disposition is WorkerRecoveryOrchestrationDisposition.RECONCILIATION_REQUIRED
    assert ctx["dispatch"].calls == []
    assert result.episode.attempt_count == 1


@pytest.mark.asyncio
async def test_crash_after_bind_execution_in_progress_zero_dispatch() -> None:
    execution_id = mint_execution_id()
    reader = StubExecutionOutcomeReader(CanonicalExecutionTerminalDisposition.IN_PROGRESS)
    service, ctx = _harness()
    service._execution_outcome_reader = reader
    request = _orchestration_request()
    episode_id = derive_recovery_episode_id(
        worker_instance_id=_WORKER_ID,
        obstacle_id=request.decision.obstacle_id,
        recovery_decision_id=request.decision.decision_id,
    )
    from intergrax.contracts.autonomous_work.obstacle_recovery import DECISION_POLICY_VERSION
    from intergrax.contracts.autonomous_work.recovery_orchestration import WorkerRecoveryEpisode
    from intergrax.contracts.autonomous_work.revision import initial_revision

    bound = WorkerRecoveryEpisode(
        recovery_episode_id=episode_id,
        worker_instance_id=_WORKER_ID,
        obstacle_id=request.decision.obstacle_id,
        recovery_decision_id=request.decision.decision_id,
        decision_policy_version=DECISION_POLICY_VERSION,
        strategy=request.decision.strategy,
        original_source=request.original_source,
        resume_target=request.resume_target,
        started_at=_NOW,
        status=RecoveryEpisodeStatus.IN_PROGRESS,
        attempt_count=1,
        revision=Revision(2),
        max_attempts=2,
        claimed_attempt_number=1,
        last_execution_id=execution_id,
        pre_recovery_lifecycle_state=WorkerLifecycleState.WORKING,
    )
    ctx["episode_repo"].create_or_get(
        replace(bound, status=RecoveryEpisodeStatus.PENDING, attempt_count=0, revision=initial_revision()),
    )
    ctx["episode_repo"]._records[episode_id] = bound  # type: ignore[attr-defined]
    result = await service.orchestrate(request, dispatch_request=_dispatch_request())
    assert result.disposition is WorkerRecoveryOrchestrationDisposition.ATTEMPT_DISPATCHED
    assert ctx["dispatch"].calls == []


@pytest.mark.asyncio
async def test_crash_after_bind_execution_success_resumes() -> None:
    execution_id = mint_execution_id()
    reader = StubExecutionOutcomeReader(CanonicalExecutionTerminalDisposition.SUCCEEDED)
    service, ctx = _harness()
    service._execution_outcome_reader = reader
    request = _orchestration_request()
    episode_id = derive_recovery_episode_id(
        worker_instance_id=_WORKER_ID,
        obstacle_id=request.decision.obstacle_id,
        recovery_decision_id=request.decision.decision_id,
    )
    from intergrax.contracts.autonomous_work.obstacle_recovery import DECISION_POLICY_VERSION
    from intergrax.contracts.autonomous_work.recovery_orchestration import WorkerRecoveryEpisode

    seed = WorkerRecoveryEpisode(
        recovery_episode_id=episode_id,
        worker_instance_id=_WORKER_ID,
        obstacle_id=request.decision.obstacle_id,
        recovery_decision_id=request.decision.decision_id,
        decision_policy_version=DECISION_POLICY_VERSION,
        strategy=request.decision.strategy,
        original_source=request.original_source,
        resume_target=request.resume_target,
        started_at=_NOW,
        status=RecoveryEpisodeStatus.IN_PROGRESS,
        attempt_count=1,
        revision=Revision(1),
        max_attempts=2,
        claimed_attempt_number=1,
        last_execution_id=execution_id,
        pre_recovery_lifecycle_state=WorkerLifecycleState.WORKING,
    )
    ctx["episode_repo"].create_or_get(seed)
    result = await service.orchestrate(request, dispatch_request=_dispatch_request())
    assert result.disposition is WorkerRecoveryOrchestrationDisposition.RESUMED
    assert result.resume_intent is not None
    assert result.episode.status is RecoveryEpisodeStatus.SUCCEEDED
    assert ctx["dispatch"].calls == []


@pytest.mark.asyncio
async def test_quarantine_lifecycle_conflict_does_not_mark_episode() -> None:
    service, ctx = _harness()
    request = _orchestration_request(
        decision=_decision(
            strategy=RecoveryStrategy.QUARANTINE,
            decision_reason_code=RecoveryDecisionReasonCode.SUSPICIOUS_QUARANTINE,
            max_attempts=None,
            obstacle_kind=WorkerObstacleKind.SUSPICIOUS_OR_UNSAFE,
        ),
    )
    conflict = AutonomousWorkRevisionConflict(
        "conflict",
        entity_kind="WorkerInstance",
        entity_id=_WORKER_ID,
        expected_revision=Revision(0),
        actual_revision=Revision(1),
    )
    with patch.object(
        service._lifecycle_service,
        "transition",
        side_effect=conflict,
    ):
        result = await service.orchestrate(request)
    assert result.disposition is WorkerRecoveryOrchestrationDisposition.CONFLICT
    assert result.episode.status is RecoveryEpisodeStatus.PENDING
    worker = ctx["worker_repo"].get(worker_instance_id=_WORKER_ID)
    assert worker is not None
    assert worker.lifecycle_state is WorkerLifecycleState.WORKING


@pytest.mark.asyncio
async def test_partial_resume_continuity_then_lifecycle_conflict() -> None:
    service, ctx = _harness()
    continuity = ctx["continuity_repo"].get(worker_instance_id=_WORKER_ID)
    assert continuity is not None
    continuity = replace(
        continuity,
        unresolved_problem_refs=(_EVIDENCE_REF,),
    )
    ctx["continuity_repo"].replace(continuity, expected_revision=continuity.revision)
    continuity = ctx["continuity_repo"].get(worker_instance_id=_WORKER_ID)
    assert continuity is not None
    request = _orchestration_request(continuity_expected_revision=continuity.revision)
    conflict = AutonomousWorkRevisionConflict(
        "conflict",
        entity_kind="WorkerInstance",
        entity_id=_WORKER_ID,
        expected_revision=Revision(0),
        actual_revision=Revision(1),
    )
    continuity_replace_calls = 0
    original_replace = ctx["continuity_repo"].replace
    original_lifecycle_transition = service._lifecycle_service.transition
    lifecycle_transition_calls = 0

    def counting_replace(updated, *, expected_revision):
        nonlocal continuity_replace_calls
        continuity_replace_calls += 1
        return original_replace(updated, expected_revision=expected_revision)

    def lifecycle_conflict_on_resume(request):
        nonlocal lifecycle_transition_calls
        lifecycle_transition_calls += 1
        if request.target_state is WorkerLifecycleState.WORKING:
            raise conflict
        return original_lifecycle_transition(request)

    with (
        patch.object(ctx["continuity_repo"], "replace", side_effect=counting_replace),
        patch.object(
            service._lifecycle_service,
            "transition",
            side_effect=lifecycle_conflict_on_resume,
        ),
    ):
        first = await service.orchestrate(request, dispatch_request=_dispatch_request())

    assert first.disposition is WorkerRecoveryOrchestrationDisposition.CONFLICT
    assert first.resume_intent is None
    assert first.episode.status is RecoveryEpisodeStatus.IN_PROGRESS
    assert first.episode.continuity_resume_completed is True
    assert first.episode.continuity_resume_revision is not None
    assert continuity_replace_calls == 1

    second = await service.orchestrate(request, dispatch_request=_dispatch_request())
    assert second.disposition is WorkerRecoveryOrchestrationDisposition.RESUMED
    assert second.resume_intent is not None
    assert second.resume_intent.continuity_revision == first.episode.continuity_resume_revision
    assert second.episode.status is RecoveryEpisodeStatus.SUCCEEDED
    assert continuity_replace_calls == 1


@pytest.mark.asyncio
async def test_partial_resume_survives_restart_between_continuity_and_lifecycle() -> None:
    service, ctx = _harness()
    continuity = ctx["continuity_repo"].get(worker_instance_id=_WORKER_ID)
    assert continuity is not None
    continuity = replace(
        continuity,
        unresolved_problem_refs=(_EVIDENCE_REF,),
    )
    ctx["continuity_repo"].replace(continuity, expected_revision=continuity.revision)
    continuity = ctx["continuity_repo"].get(worker_instance_id=_WORKER_ID)
    assert continuity is not None
    request = _orchestration_request(continuity_expected_revision=continuity.revision)
    conflict = AutonomousWorkRevisionConflict(
        "conflict",
        entity_kind="WorkerInstance",
        entity_id=_WORKER_ID,
        expected_revision=Revision(0),
        actual_revision=Revision(1),
    )
    original_lifecycle_transition = service._lifecycle_service.transition

    def lifecycle_conflict_on_resume(request):
        if request.target_state is WorkerLifecycleState.WORKING:
            raise conflict
        return original_lifecycle_transition(request)

    with patch.object(
        service._lifecycle_service,
        "transition",
        side_effect=lifecycle_conflict_on_resume,
    ):
        first = await service.orchestrate(request, dispatch_request=_dispatch_request())

    assert first.episode.continuity_resume_completed is True
    persisted_episode = first.episode
    persisted_continuity = ctx["continuity_repo"].get(worker_instance_id=_WORKER_ID)

    service2, ctx2 = _harness()
    ctx2["episode_repo"].create_or_get(
        replace(persisted_episode, status=RecoveryEpisodeStatus.PENDING, attempt_count=0),
    )
    ctx2["episode_repo"]._records[persisted_episode.recovery_episode_id] = persisted_episode  # type: ignore[attr-defined]
    assert persisted_continuity is not None
    ctx2["continuity_repo"]._store._records[_WORKER_ID] = persisted_continuity  # type: ignore[attr-defined]

    continuity_replace_calls = 0
    original_replace = ctx2["continuity_repo"].replace

    def counting_replace(updated, *, expected_revision):
        nonlocal continuity_replace_calls
        continuity_replace_calls += 1
        return original_replace(updated, expected_revision=expected_revision)

    with patch.object(ctx2["continuity_repo"], "replace", side_effect=counting_replace):
        second = await service2.orchestrate(request, dispatch_request=_dispatch_request())

    assert second.disposition is WorkerRecoveryOrchestrationDisposition.RESUMED
    assert second.episode.status is RecoveryEpisodeStatus.SUCCEEDED
    assert second.resume_intent is not None
    assert second.resume_intent.continuity_revision == persisted_episode.continuity_resume_revision
    assert continuity_replace_calls == 0


# © Artur Czarnecki. All rights reserved.

"""AW-4B — bounded proactive WorkerGoal evaluation tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.autonomous_work.goal_evaluation_ports import (
    DeterministicThresholdGoalEvaluator,
    InMemoryGoalEvaluationCadenceStateStore,
    MappingGoalEvaluationCadenceResolver,
    MappingGoalProgressProjectionResolver,
)
from intergrax.autonomous_work.goal_evaluation_service import (
    WorkerGoalEvaluationService,
)
from intergrax.autonomous_work.in_memory_repository import (
    InMemoryResponsibilityRepository,
    InMemoryWorkerGoalRepository,
    InMemoryWorkContinuityStateRepository,
    InMemoryWorkerInstanceRepository,
    InMemoryWorkerWakeUpReceiptRepository,
)
from intergrax.autonomous_work.wake_up_service import WorkerWakeUpService
from intergrax.contracts.autonomous_work import (
    GoalEvaluationDisposition,
    GoalEvaluationReasonCode,
    GoalProgressProjection,
    WorkerGoalEvaluationRequest,
    WorkerGoalStatus,
    WorkerLifecycleState,
    WorkerWakeUpContext,
    WorkerWakeUpDisposition,
    WorkerWakeUpSourceKind,
    mint_wake_up_id,
    mint_worker_goal_id,
)
from intergrax.contracts.autonomous_work.references import (
    EvaluationCadenceRef,
    ProgressProjectionRef,
    WakeUpSourceRef,
    WorkReference,
)
from intergrax.contracts.autonomous_work.wake_up import WorkerWakeUpSignal
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = pytest.mark.unit

_UTC = UTC
_NOW = datetime(2026, 9, 3, 12, 0, tzinfo=_UTC)
_CADENCE_REF = EvaluationCadenceRef("cadence/goal-eval-5m")
_CADENCE_INTERVAL_SECONDS = 300
_PROJECTION_REF = ProgressProjectionRef("projection/sla-30m")


def _evaluation_service(
    *,
    cadence_store: InMemoryGoalEvaluationCadenceStateStore | None = None,
    projections: dict[ProgressProjectionRef, GoalProgressProjection] | None = None,
) -> tuple[
    WorkerGoalEvaluationService,
    InMemoryResponsibilityRepository,
    InMemoryWorkerGoalRepository,
    InMemoryGoalEvaluationCadenceStateStore,
]:
    responsibility_repo = InMemoryResponsibilityRepository()
    goal_repo = InMemoryWorkerGoalRepository()
    cadence_store = cadence_store or InMemoryGoalEvaluationCadenceStateStore()
    projections = (
        {
            _PROJECTION_REF: GoalProgressProjection(
                projection_ref=_PROJECTION_REF,
                observed_at=_NOW,
                current_value=1.0,
                target_value=1.0,
                status="healthy",
                evidence_refs=("evidence/progress/healthy",),
            )
        }
        if projections is None
        else projections
    )
    service = WorkerGoalEvaluationService(
        responsibility_repository=responsibility_repo,
        worker_goal_repository=goal_repo,
        cadence_resolver=MappingGoalEvaluationCadenceResolver(
            {_CADENCE_REF: _CADENCE_INTERVAL_SECONDS}
        ),
        progress_projection_resolver=MappingGoalProgressProjectionResolver(projections),
        goal_evaluator=DeterministicThresholdGoalEvaluator(),
        cadence_state=cadence_store,
    )
    return service, responsibility_repo, goal_repo, cadence_store


def _accepted_context(
    *,
    worker_id: str,
    lifecycle_state: WorkerLifecycleState = WorkerLifecycleState.IDLE,
) -> WorkerWakeUpContext:
    worker = contract_suite.worker_instance(
        worker_instance_id=worker_id,
        lifecycle_state=lifecycle_state,
    )
    wake_up_id = mint_wake_up_id()
    signal = WorkerWakeUpSignal(
        wake_up_id=wake_up_id,
        worker_instance_id=worker_id,
        source_kind=WorkerWakeUpSourceKind.SCHEDULE,
        source_ref=WakeUpSourceRef("schedule/goal-eval"),
        occurred_at=_NOW - timedelta(minutes=1),
        delivery_identity=wake_up_id,
    )
    return WorkerWakeUpContext(
        worker_instance=worker,
        wake_up_signal=signal,
        continuity_state=None,
        accepted_at=_NOW,
        disposition=WorkerWakeUpDisposition.ACCEPTED,
        receipt=None,
    )


def _seed_goal_chain(
    *,
    responsibility_repo: InMemoryResponsibilityRepository,
    goal_repo: InMemoryWorkerGoalRepository,
    worker_id: str,
    **goal_overrides: object,
) -> tuple[str, str]:
    responsibility = contract_suite.responsibility(worker_instance_id=worker_id)
    responsibility_repo.create(responsibility)
    cadence_ref = goal_overrides.pop("evaluation_cadence_ref", _CADENCE_REF)
    goal = contract_suite.worker_goal(
        responsibility_id=responsibility.responsibility_id,
        evaluation_cadence_ref=cadence_ref,
        progress_projection_ref=_PROJECTION_REF,
        **goal_overrides,
    )
    goal_repo.create(goal)
    return responsibility.responsibility_id, goal.goal_id


def test_not_due_skips_evaluator() -> None:
    service, responsibility_repo, goal_repo, cadence_store = _evaluation_service()
    worker_id = contract_suite.worker_instance().worker_instance_id
    _, goal_id = _seed_goal_chain(
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
        worker_id=worker_id,
    )
    cadence_store.record_evaluated(goal_id=goal_id, evaluated_at=_NOW - timedelta(minutes=1))
    result = service.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=_accepted_context(worker_id=worker_id),
            evaluated_at=_NOW,
            max_goals=10,
        )
    )
    assert len(result.decisions) == 1
    decision = result.decisions[0]
    assert decision.disposition == GoalEvaluationDisposition.NOT_DUE
    assert result.goals_skipped_not_due == 1
    assert result.goals_evaluated == 0


def test_due_healthy_goal_returns_no_action_with_evidence() -> None:
    service, responsibility_repo, goal_repo, _ = _evaluation_service()
    worker_id = contract_suite.worker_instance().worker_instance_id
    _seed_goal_chain(
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
        worker_id=worker_id,
    )
    result = service.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=_accepted_context(worker_id=worker_id),
            evaluated_at=_NOW,
            max_goals=10,
        )
    )
    decision = result.decisions[0]
    assert decision.disposition == GoalEvaluationDisposition.NO_ACTION
    assert decision.reason_code == GoalEvaluationReasonCode.CRITERIA_MET
    assert decision.evidence_refs


def test_due_action_required_goal_has_reason_and_evidence() -> None:
    projections = {
        _PROJECTION_REF: GoalProgressProjection(
            projection_ref=_PROJECTION_REF,
            observed_at=_NOW,
            current_value=0.2,
            target_value=0.99,
            status="at_risk",
            evidence_refs=("evidence/sla/at-risk",),
        )
    }
    service, responsibility_repo, goal_repo, _ = _evaluation_service(projections=projections)
    worker_id = contract_suite.worker_instance().worker_instance_id
    _seed_goal_chain(
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
        worker_id=worker_id,
    )
    result = service.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=_accepted_context(worker_id=worker_id),
            evaluated_at=_NOW,
            max_goals=10,
        )
    )
    decision = result.decisions[0]
    assert decision.disposition == GoalEvaluationDisposition.ACTION_REQUIRED
    assert decision.reason_code == GoalEvaluationReasonCode.SLA_RISK
    assert decision.evidence_refs == ("evidence/sla/at-risk",)


def test_suspended_completed_cancelled_goals_not_evaluated() -> None:
    service, responsibility_repo, goal_repo, _ = _evaluation_service()
    worker_id = contract_suite.worker_instance().worker_instance_id
    for status in (
        WorkerGoalStatus.SUSPENDED,
        WorkerGoalStatus.COMPLETED,
        WorkerGoalStatus.CANCELLED,
    ):
        _seed_goal_chain(
            responsibility_repo=responsibility_repo,
            goal_repo=goal_repo,
            worker_id=worker_id,
            goal_id=mint_worker_goal_id(),
            status=status,
        )
    result = service.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=_accepted_context(worker_id=worker_id),
            evaluated_at=_NOW,
            max_goals=10,
        )
    )
    assert result.decisions == ()
    assert result.goals_skipped_status == 3
    assert result.goals_evaluated == 0


def test_max_batch_evaluates_exactly_limit_in_deterministic_order() -> None:
    service, responsibility_repo, goal_repo, _ = _evaluation_service()
    worker_id = contract_suite.worker_instance().worker_instance_id
    responsibility = contract_suite.responsibility(worker_instance_id=worker_id)
    responsibility_repo.create(responsibility)
    goal_ids: list[str] = []
    for index in range(100):
        goal = contract_suite.worker_goal(
            goal_id=mint_worker_goal_id(),
            responsibility_id=responsibility.responsibility_id,
            priority=index,
            evaluation_cadence_ref=_CADENCE_REF,
            progress_projection_ref=_PROJECTION_REF,
        )
        goal_repo.create(goal)
        goal_ids.append(goal.goal_id)
    result = service.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=_accepted_context(worker_id=worker_id),
            evaluated_at=_NOW,
            max_goals=10,
        )
    )
    assert result.goals_considered == 100
    assert result.goals_evaluated == 10
    assert result.goals_skipped_batch_limit == 90
    evaluated_ids = [decision.goal_id for decision in result.decisions]
    all_goals = [goal_repo.get(goal_id=goal_id) for goal_id in goal_ids]
    all_goals = [goal for goal in all_goals if goal is not None]
    expected_ids = [
        goal.goal_id
        for goal in sorted(all_goals, key=lambda goal: (-goal.priority, goal.goal_id))[:10]
    ]
    assert evaluated_ids == expected_ids


def test_deterministic_order_independent_of_insertion_order() -> None:
    worker_id = contract_suite.worker_instance().worker_instance_id
    responsibility = contract_suite.responsibility(worker_instance_id=worker_id)
    goals = [
        contract_suite.worker_goal(
            goal_id=mint_worker_goal_id(),
            responsibility_id=responsibility.responsibility_id,
            priority=priority,
            evaluation_cadence_ref=_CADENCE_REF,
            progress_projection_ref=_PROJECTION_REF,
        )
        for priority in (5, 1, 9, 3)
    ]

    service_a, responsibility_repo_a, goal_repo_a, _ = _evaluation_service()
    responsibility_repo_a.create(responsibility)
    for goal in goals:
        goal_repo_a.create(goal)
    result_a = service_a.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=_accepted_context(worker_id=worker_id),
            evaluated_at=_NOW,
            max_goals=10,
        )
    )

    service_b, responsibility_repo_b, goal_repo_b, _ = _evaluation_service()
    responsibility_repo_b.create(responsibility)
    for goal in reversed(goals):
        goal_repo_b.create(goal)
    result_b = service_b.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=_accepted_context(worker_id=worker_id),
            evaluated_at=_NOW,
            max_goals=10,
        )
    )
    assert [decision.goal_id for decision in result_a.decisions] == [
        decision.goal_id for decision in result_b.decisions
    ]


def test_duplicate_wake_up_does_not_evaluate_via_helper() -> None:
    wake_service, worker_repo, continuity_repo, receipt_repo = _wake_services()
    worker_id = contract_suite.worker_instance(
        lifecycle_state=WorkerLifecycleState.IDLE,
    ).worker_instance_id
    worker_repo.create(
        contract_suite.worker_instance(
            worker_instance_id=worker_id,
            lifecycle_state=WorkerLifecycleState.IDLE,
        )
    )
    wake_id = mint_wake_up_id()
    signal = WorkerWakeUpSignal(
        wake_up_id=wake_id,
        worker_instance_id=worker_id,
        source_kind=WorkerWakeUpSourceKind.SCHEDULE,
        source_ref=WakeUpSourceRef("schedule/goal-eval"),
        occurred_at=_NOW,
        delivery_identity=wake_id,
    )
    first = wake_service.accept(signal)
    second = wake_service.accept(signal)
    eval_service, responsibility_repo, goal_repo, _ = _evaluation_service()
    _seed_goal_chain(
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
        worker_id=worker_id,
    )
    assert eval_service.evaluate_from_wake_up_result(first, max_goals=10) is not None
    assert eval_service.evaluate_from_wake_up_result(second, max_goals=10) is None


def test_conflict_wake_up_does_not_evaluate() -> None:
    wake_service, worker_repo, _, _ = _wake_services()
    worker = contract_suite.worker_instance(lifecycle_state=WorkerLifecycleState.IDLE)
    worker_repo.create(worker)
    wake_id = mint_wake_up_id()
    first = wake_service.accept(
        WorkerWakeUpSignal(
            wake_up_id=wake_id,
            worker_instance_id=worker.worker_instance_id,
            source_kind=WorkerWakeUpSourceKind.SCHEDULE,
            source_ref=WakeUpSourceRef("schedule/a"),
            occurred_at=_NOW,
            delivery_identity=mint_wake_up_id(),
        )
    )
    conflict = wake_service.accept(
        WorkerWakeUpSignal(
            wake_up_id=wake_id,
            worker_instance_id=worker.worker_instance_id,
            source_kind=WorkerWakeUpSourceKind.OPERATOR,
            source_ref=WakeUpSourceRef("operator/b"),
            occurred_at=_NOW,
            delivery_identity=mint_wake_up_id(),
        )
    )
    eval_service, _, _, _ = _evaluation_service()
    assert first.disposition == WorkerWakeUpDisposition.ACCEPTED
    assert conflict.disposition == WorkerWakeUpDisposition.CONFLICT
    assert eval_service.evaluate_from_wake_up_result(conflict, max_goals=10) is None


def test_cross_worker_goal_isolation() -> None:
    service, responsibility_repo, goal_repo, _ = _evaluation_service()
    worker_a = contract_suite.worker_instance().worker_instance_id
    worker_b = contract_suite.worker_instance().worker_instance_id
    _seed_goal_chain(
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
        worker_id=worker_a,
    )
    other_responsibility = contract_suite.responsibility(worker_instance_id=worker_b)
    responsibility_repo.create(other_responsibility)
    goal_repo.create(
        contract_suite.worker_goal(
            responsibility_id=other_responsibility.responsibility_id,
            evaluation_cadence_ref=_CADENCE_REF,
            progress_projection_ref=_PROJECTION_REF,
        )
    )
    result = service.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=_accepted_context(worker_id=worker_a),
            evaluated_at=_NOW,
            max_goals=10,
        )
    )
    assert result.goals_evaluated == 1
    assert len(result.decisions) == 1


def test_missing_projection_is_not_evaluable() -> None:
    service, responsibility_repo, goal_repo, _ = _evaluation_service(projections={})
    worker_id = contract_suite.worker_instance().worker_instance_id
    _seed_goal_chain(
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
        worker_id=worker_id,
    )
    result = service.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=_accepted_context(worker_id=worker_id),
            evaluated_at=_NOW,
            max_goals=10,
        )
    )
    decision = result.decisions[0]
    assert decision.disposition == GoalEvaluationDisposition.NOT_EVALUABLE
    assert decision.reason_code == GoalEvaluationReasonCode.PROGRESS_PROJECTION_UNAVAILABLE


def test_paused_worker_returns_empty_batch() -> None:
    service, responsibility_repo, goal_repo, _ = _evaluation_service()
    worker_id = contract_suite.worker_instance().worker_instance_id
    _seed_goal_chain(
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
        worker_id=worker_id,
    )
    result = service.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=_accepted_context(
                worker_id=worker_id,
                lifecycle_state=WorkerLifecycleState.PAUSED,
            ),
            evaluated_at=_NOW,
            max_goals=10,
        )
    )
    assert result.decisions == ()
    assert result.goals_evaluated == 0


def test_non_accepted_request_rejected_at_contract_boundary() -> None:
    worker_id = contract_suite.worker_instance().worker_instance_id
    context = replace(
        _accepted_context(worker_id=worker_id),
        disposition=WorkerWakeUpDisposition.DUPLICATE,
    )
    with pytest.raises(ValueError, match="ACCEPTED"):
        WorkerGoalEvaluationRequest(
            wake_up_context=context,
            evaluated_at=_NOW,
            max_goals=10,
        )


def test_not_evaluable_records_cadence_to_prevent_hot_loop() -> None:
    service, responsibility_repo, goal_repo, cadence_store = _evaluation_service(projections={})
    worker_id = contract_suite.worker_instance().worker_instance_id
    _, goal_id = _seed_goal_chain(
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
        worker_id=worker_id,
    )
    service.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=_accepted_context(worker_id=worker_id),
            evaluated_at=_NOW,
            max_goals=10,
        )
    )
    assert cadence_store.last_evaluated_at(goal_id=goal_id) == _NOW
    result = service.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=_accepted_context(worker_id=worker_id),
            evaluated_at=_NOW + timedelta(minutes=1),
            max_goals=10,
        )
    )
    assert result.decisions[0].disposition == GoalEvaluationDisposition.NOT_DUE


def test_five_minute_cadence_boundedness() -> None:
    service, responsibility_repo, goal_repo, cadence_store = _evaluation_service()
    worker_id = contract_suite.worker_instance().worker_instance_id
    _, goal_id = _seed_goal_chain(
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
        worker_id=worker_id,
    )
    first = service.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=_accepted_context(worker_id=worker_id),
            evaluated_at=_NOW,
            max_goals=10,
        )
    )
    assert first.decisions[0].disposition == GoalEvaluationDisposition.NO_ACTION
    second = service.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=_accepted_context(worker_id=worker_id),
            evaluated_at=_NOW + timedelta(minutes=1),
            max_goals=10,
        )
    )
    assert second.decisions[0].disposition == GoalEvaluationDisposition.NOT_DUE
    third = service.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=_accepted_context(worker_id=worker_id),
            evaluated_at=_NOW + timedelta(minutes=5),
            max_goals=10,
        )
    )
    assert third.decisions[0].disposition == GoalEvaluationDisposition.NO_ACTION
    assert cadence_store.last_evaluated_at(goal_id=goal_id) == _NOW + timedelta(minutes=5)


def test_cadence_survives_restart_with_new_store_instance() -> None:
    worker_id = contract_suite.worker_instance().worker_instance_id
    cadence_store = InMemoryGoalEvaluationCadenceStateStore()
    service, responsibility_repo, goal_repo, _ = _evaluation_service(cadence_store=cadence_store)
    _, goal_id = _seed_goal_chain(
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
        worker_id=worker_id,
    )
    service.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=_accepted_context(worker_id=worker_id),
            evaluated_at=_NOW,
            max_goals=10,
        )
    )
    restarted_service = WorkerGoalEvaluationService(
        responsibility_repository=responsibility_repo,
        worker_goal_repository=goal_repo,
        cadence_resolver=MappingGoalEvaluationCadenceResolver(
            {_CADENCE_REF: _CADENCE_INTERVAL_SECONDS}
        ),
        progress_projection_resolver=MappingGoalProgressProjectionResolver(
            {
                _PROJECTION_REF: GoalProgressProjection(
                    projection_ref=_PROJECTION_REF,
                    observed_at=_NOW,
                    current_value=1.0,
                    target_value=1.0,
                    status="healthy",
                    evidence_refs=("evidence/progress/healthy",),
                )
            }
        ),
        goal_evaluator=DeterministicThresholdGoalEvaluator(),
        cadence_state=cadence_store,
    )
    not_due = restarted_service.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=_accepted_context(worker_id=worker_id),
            evaluated_at=_NOW + timedelta(minutes=4),
            max_goals=10,
        )
    )
    assert not_due.decisions[0].disposition == GoalEvaluationDisposition.NOT_DUE
    due = restarted_service.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=_accepted_context(worker_id=worker_id),
            evaluated_at=_NOW + timedelta(hours=1),
            max_goals=10,
        )
    )
    assert due.decisions[0].disposition == GoalEvaluationDisposition.NO_ACTION
    assert cadence_store.last_evaluated_at(goal_id=goal_id) == _NOW + timedelta(hours=1)


def test_unknown_cadence_ref_fails_closed_as_not_evaluable() -> None:
    service, responsibility_repo, goal_repo, cadence_store = _evaluation_service()
    worker_id = contract_suite.worker_instance().worker_instance_id
    _, goal_id = _seed_goal_chain(
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
        worker_id=worker_id,
        evaluation_cadence_ref=EvaluationCadenceRef("cadence/unknown"),
    )
    result = service.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=_accepted_context(worker_id=worker_id),
            evaluated_at=_NOW,
            max_goals=10,
        )
    )
    decision = result.decisions[0]
    assert decision.disposition == GoalEvaluationDisposition.NOT_EVALUABLE
    assert decision.reason_code == GoalEvaluationReasonCode.CADENCE_POLICY_UNAVAILABLE
    assert cadence_store.last_evaluated_at(goal_id=goal_id) == _NOW


def test_unrelated_open_work_does_not_suppress_action_required() -> None:
    worker_id = contract_suite.worker_instance().worker_instance_id
    responsibility = contract_suite.responsibility(worker_instance_id=worker_id)
    goal_a = contract_suite.worker_goal(
        goal_id=mint_worker_goal_id(),
        responsibility_id=responsibility.responsibility_id,
        evaluation_cadence_ref=_CADENCE_REF,
        progress_projection_ref=_PROJECTION_REF,
    )
    goal_b = contract_suite.worker_goal(
        goal_id=mint_worker_goal_id(),
        responsibility_id=responsibility.responsibility_id,
        evaluation_cadence_ref=_CADENCE_REF,
        progress_projection_ref=_PROJECTION_REF,
    )
    continuity = contract_suite.continuity_state(
        worker_instance_ref=worker_id,
        active_goal_refs=(goal_a.goal_id, goal_b.goal_id),
        open_work_refs=(WorkReference("work/for-goal-b"),),
    )
    projections = {
        _PROJECTION_REF: GoalProgressProjection(
            projection_ref=_PROJECTION_REF,
            observed_at=_NOW,
            current_value=0.2,
            target_value=0.99,
            status="at_risk",
            evidence_refs=("evidence/sla/at-risk",),
        )
    }
    service, responsibility_repo, goal_repo, _ = _evaluation_service(projections=projections)
    responsibility_repo.create(responsibility)
    goal_repo.create(goal_a)
    goal_repo.create(goal_b)
    context = replace(_accepted_context(worker_id=worker_id), continuity_state=continuity)
    result = service.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=context,
            evaluated_at=_NOW,
            max_goals=10,
        )
    )
    goal_a_decision = next(
        decision for decision in result.decisions if decision.goal_id == goal_a.goal_id
    )
    assert goal_a_decision.disposition == GoalEvaluationDisposition.ACTION_REQUIRED
    assert goal_a_decision.reason_code == GoalEvaluationReasonCode.SLA_RISK


def test_open_work_continuity_does_not_suppress_without_goal_work_correlation() -> None:
    """AW-4C owns Goal↔Work correlation; AW-4B must not infer it from open_work_refs."""
    worker_id = contract_suite.worker_instance().worker_instance_id
    responsibility = contract_suite.responsibility(worker_instance_id=worker_id)
    goal = contract_suite.worker_goal(
        responsibility_id=responsibility.responsibility_id,
        evaluation_cadence_ref=_CADENCE_REF,
        progress_projection_ref=_PROJECTION_REF,
    )
    continuity = contract_suite.continuity_state(
        worker_instance_ref=worker_id,
        active_goal_refs=(goal.goal_id,),
        open_work_refs=(WorkReference("work/open-1"),),
    )
    projections = {
        _PROJECTION_REF: GoalProgressProjection(
            projection_ref=_PROJECTION_REF,
            observed_at=_NOW,
            current_value=0.2,
            target_value=0.99,
            status="breached",
            evidence_refs=("evidence/threshold/breached",),
        )
    }
    service, responsibility_repo, goal_repo, _ = _evaluation_service(projections=projections)
    responsibility_repo.create(responsibility)
    goal_repo.create(goal)
    context = replace(_accepted_context(worker_id=worker_id), continuity_state=continuity)
    result = service.evaluate(
        WorkerGoalEvaluationRequest(
            wake_up_context=context,
            evaluated_at=_NOW,
            max_goals=10,
        )
    )
    decision = result.decisions[0]
    assert decision.disposition == GoalEvaluationDisposition.ACTION_REQUIRED
    assert decision.reason_code == GoalEvaluationReasonCode.THRESHOLD_BREACH


def _wake_services() -> tuple[
    WorkerWakeUpService,
    InMemoryWorkerInstanceRepository,
    InMemoryWorkContinuityStateRepository,
    InMemoryWorkerWakeUpReceiptRepository,
]:
    worker_repo = InMemoryWorkerInstanceRepository()
    continuity_repo = InMemoryWorkContinuityStateRepository()
    receipt_repo = InMemoryWorkerWakeUpReceiptRepository()

    class _Clock:
        def now(self) -> datetime:
            return _NOW

    wake_service = WorkerWakeUpService(
        worker_instance_repository=worker_repo,
        continuity_state_repository=continuity_repo,
        wake_up_receipt_repository=receipt_repo,
        clock=_Clock(),
    )
    return wake_service, worker_repo, continuity_repo, receipt_repo

# © Artur Czarnecki. All rights reserved.

"""AW-4C — collaborative work bridge tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from intergrax.autonomous_work.collaborative_work_intake import (
    RecordingCollaborativeWorkIntake,
    UnavailableCollaborativeWorkIntake,
)
from intergrax.autonomous_work.in_memory_repository import (
    InMemoryResponsibilityRepository,
    InMemoryWorkerGoalRepository,
    InMemoryWorkerInstanceRepository,
)
from intergrax.autonomous_work.worker_collaborative_work_bridge import (
    WorkerCollaborativeWorkBridge,
    WorkerCollaborativeWorkBridgeRejected,
)
from intergrax.contracts.autonomous_work import (
    GoalEvaluationDecision,
    GoalEvaluationDisposition,
    GoalEvaluationReasonCode,
    ResponsibilityStatus,
    WorkerGoalStatus,
    WorkerLifecycleState,
    mint_wake_up_id,
)
from intergrax.contracts.autonomous_work.collaborative_work_bridge import (
    CollaborativeWorkBridgeRejectionReason,
    CollaborativeWorkSubmissionDisposition,
    derive_collaborative_work_request_identity,
)
from intergrax.contracts.autonomous_work.references import ProgressProjectionRef
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = pytest.mark.unit

_UTC = UTC
_NOW = datetime(2026, 9, 4, 12, 0, tzinfo=_UTC)
_PROJECTION_REF = ProgressProjectionRef("projection/sla-30m")


def _bridge(
    intake: RecordingCollaborativeWorkIntake | UnavailableCollaborativeWorkIntake | None = None,
) -> tuple[
    WorkerCollaborativeWorkBridge,
    InMemoryWorkerInstanceRepository,
    InMemoryResponsibilityRepository,
    InMemoryWorkerGoalRepository,
    RecordingCollaborativeWorkIntake,
]:
    worker_repo = InMemoryWorkerInstanceRepository()
    responsibility_repo = InMemoryResponsibilityRepository()
    goal_repo = InMemoryWorkerGoalRepository()
    recording_intake = (
        intake if isinstance(intake, RecordingCollaborativeWorkIntake) else RecordingCollaborativeWorkIntake()
    )
    bridge = WorkerCollaborativeWorkBridge(
        worker_instance_repository=worker_repo,
        responsibility_repository=responsibility_repo,
        worker_goal_repository=goal_repo,
        intake_port=recording_intake,
        clock=lambda: _NOW,
    )
    return bridge, worker_repo, responsibility_repo, goal_repo, recording_intake


def _seed_active_chain(
    *,
    worker_repo: InMemoryWorkerInstanceRepository,
    responsibility_repo: InMemoryResponsibilityRepository,
    goal_repo: InMemoryWorkerGoalRepository,
    worker_id: str | None = None,
    lifecycle_state: WorkerLifecycleState = WorkerLifecycleState.IDLE,
    goal_status: WorkerGoalStatus = WorkerGoalStatus.ACTIVE,
    responsibility_status: ResponsibilityStatus = ResponsibilityStatus.ACTIVE,
) -> tuple[str, str, str]:
    worker = contract_suite.worker_instance(
        lifecycle_state=lifecycle_state,
        **({"worker_instance_id": worker_id} if worker_id is not None else {}),
    )
    worker_repo.create(worker)
    responsibility = contract_suite.responsibility(
        worker_instance_id=worker.worker_instance_id,
        status=responsibility_status,
    )
    responsibility_repo.create(responsibility)
    goal = contract_suite.worker_goal(
        responsibility_id=responsibility.responsibility_id,
        status=goal_status,
        progress_projection_ref=_PROJECTION_REF,
    )
    goal_repo.create(goal)
    return worker.worker_instance_id, responsibility.responsibility_id, goal.goal_id


def _action_required_decision(
    *,
    goal_id: str,
    wake_up_id: str | None = None,
) -> GoalEvaluationDecision:
    return GoalEvaluationDecision(
        goal_id=goal_id,
        evaluated_at=_NOW,
        disposition=GoalEvaluationDisposition.ACTION_REQUIRED,
        reason="SLA risk requires collaborative work",
        wake_up_id=wake_up_id or mint_wake_up_id(),
        reason_code=GoalEvaluationReasonCode.SLA_RISK,
        evidence_refs=("evidence/sla/at-risk",),
        progress_projection_ref=_PROJECTION_REF,
    )


def _no_action_decision(goal_id: str) -> GoalEvaluationDecision:
    return GoalEvaluationDecision(
        goal_id=goal_id,
        evaluated_at=_NOW,
        disposition=GoalEvaluationDisposition.NO_ACTION,
        reason="criteria met",
        wake_up_id=mint_wake_up_id(),
        reason_code=GoalEvaluationReasonCode.CRITERIA_MET,
    )


def test_action_required_submits_one_request_with_provenance() -> None:
    bridge, worker_repo, responsibility_repo, goal_repo, intake = _bridge()
    worker_id, responsibility_id, goal_id = _seed_active_chain(
        worker_repo=worker_repo,
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
    )
    decision = _action_required_decision(goal_id=goal_id)
    result = bridge.submit_from_decision(
        worker_instance_id=worker_id,
        decision=decision,
    )
    assert result.disposition == CollaborativeWorkSubmissionDisposition.ACCEPTED
    assert len(intake.submissions) == 1
    request = intake.submissions[0]
    assert request.worker_instance_id == worker_id
    assert request.responsibility_id == responsibility_id
    assert request.goal_id == goal_id
    assert request.wake_up_id == decision.wake_up_id
    assert request.reason_code == GoalEvaluationReasonCode.SLA_RISK
    assert request.evidence_refs == ("evidence/sla/at-risk",)
    assert request.progress_projection_ref == _PROJECTION_REF
    assert request.requested_priority == contract_suite.worker_goal().priority
    assert request.title == contract_suite.worker_goal(
        responsibility_id=responsibility_id,
        progress_projection_ref=_PROJECTION_REF,
    ).objective


@pytest.mark.parametrize(
    "disposition",
    [
        GoalEvaluationDisposition.NO_ACTION,
        GoalEvaluationDisposition.NOT_DUE,
        GoalEvaluationDisposition.NOT_EVALUABLE,
    ],
)
def test_non_action_dispositions_are_structurally_rejected(
    disposition: GoalEvaluationDisposition,
) -> None:
    bridge, worker_repo, responsibility_repo, goal_repo, intake = _bridge()
    worker_id, _, goal_id = _seed_active_chain(
        worker_repo=worker_repo,
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
    )
    decision = GoalEvaluationDecision(
        goal_id=goal_id,
        evaluated_at=_NOW,
        disposition=disposition,
        reason="test",
        wake_up_id=mint_wake_up_id(),
    )
    with pytest.raises(WorkerCollaborativeWorkBridgeRejected):
        bridge.submit_from_decision(worker_instance_id=worker_id, decision=decision)
    assert intake.submissions == ()


def test_cancelled_goal_rejects_without_submission() -> None:
    bridge, worker_repo, responsibility_repo, goal_repo, intake = _bridge()
    worker_id, _, goal_id = _seed_active_chain(
        worker_repo=worker_repo,
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
        goal_status=WorkerGoalStatus.CANCELLED,
    )
    result = bridge.submit_from_decision(
        worker_instance_id=worker_id,
        decision=_action_required_decision(goal_id=goal_id),
    )
    assert result.disposition == CollaborativeWorkSubmissionDisposition.REJECTED
    assert result.rejection_reason == CollaborativeWorkBridgeRejectionReason.STALE_OR_NOT_ELIGIBLE
    assert intake.submissions == ()


def test_suspended_goal_rejects_without_submission() -> None:
    bridge, worker_repo, responsibility_repo, goal_repo, intake = _bridge()
    worker_id, _, goal_id = _seed_active_chain(
        worker_repo=worker_repo,
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
        goal_status=WorkerGoalStatus.SUSPENDED,
    )
    result = bridge.submit_from_decision(
        worker_instance_id=worker_id,
        decision=_action_required_decision(goal_id=goal_id),
    )
    assert result.disposition == CollaborativeWorkSubmissionDisposition.REJECTED
    assert result.rejection_reason == CollaborativeWorkBridgeRejectionReason.STALE_OR_NOT_ELIGIBLE
    assert intake.submissions == ()


def test_wrong_worker_rejects_without_submission() -> None:
    bridge, worker_repo, responsibility_repo, goal_repo, intake = _bridge()
    worker_id, _, goal_id = _seed_active_chain(
        worker_repo=worker_repo,
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
    )
    other_worker = contract_suite.worker_instance(
        lifecycle_state=WorkerLifecycleState.IDLE,
    )
    worker_repo.create(other_worker)
    result = bridge.submit_from_decision(
        worker_instance_id=other_worker.worker_instance_id,
        decision=_action_required_decision(goal_id=goal_id),
    )
    assert result.disposition == CollaborativeWorkSubmissionDisposition.REJECTED
    assert result.rejection_reason == CollaborativeWorkBridgeRejectionReason.OWNERSHIP_MISMATCH
    assert intake.submissions == ()


@pytest.mark.parametrize(
    "lifecycle_state",
    [
        WorkerLifecycleState.STOPPED,
        WorkerLifecycleState.QUARANTINED,
        WorkerLifecycleState.PAUSED,
        WorkerLifecycleState.PROVISIONING,
    ],
)
def test_ineligible_worker_rejects_without_submission(
    lifecycle_state: WorkerLifecycleState,
) -> None:
    bridge, worker_repo, responsibility_repo, goal_repo, intake = _bridge()
    worker_id, _, goal_id = _seed_active_chain(
        worker_repo=worker_repo,
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
        lifecycle_state=lifecycle_state,
    )
    result = bridge.submit_from_decision(
        worker_instance_id=worker_id,
        decision=_action_required_decision(goal_id=goal_id),
    )
    assert result.disposition == CollaborativeWorkSubmissionDisposition.REJECTED
    assert result.rejection_reason == CollaborativeWorkBridgeRejectionReason.WORKER_NOT_ELIGIBLE
    assert intake.submissions == ()


def test_idempotent_retry_preserves_request_identity_and_single_submission() -> None:
    bridge, worker_repo, responsibility_repo, goal_repo, intake = _bridge()
    worker_id, _, goal_id = _seed_active_chain(
        worker_repo=worker_repo,
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
    )
    wake_up_id = mint_wake_up_id()
    decision = _action_required_decision(goal_id=goal_id, wake_up_id=wake_up_id)
    first = bridge.submit_from_decision(worker_instance_id=worker_id, decision=decision)
    second = bridge.submit_from_decision(worker_instance_id=worker_id, decision=decision)
    assert first.request_identity == second.request_identity
    assert first.request_identity == derive_collaborative_work_request_identity(
        worker_instance_id=worker_id,
        goal_id=goal_id,
        wake_up_id=wake_up_id,
    )
    assert first.disposition == CollaborativeWorkSubmissionDisposition.ACCEPTED
    assert second.disposition == CollaborativeWorkSubmissionDisposition.ALREADY_EXISTS
    assert len(intake.submissions) == 1


def test_different_wake_up_produces_distinct_request_identity() -> None:
    bridge, worker_repo, responsibility_repo, goal_repo, intake = _bridge()
    worker_id, _, goal_id = _seed_active_chain(
        worker_repo=worker_repo,
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
    )
    first_decision = _action_required_decision(goal_id=goal_id)
    second_decision = _action_required_decision(goal_id=goal_id)
    first = bridge.submit_from_decision(worker_instance_id=worker_id, decision=first_decision)
    second = bridge.submit_from_decision(worker_instance_id=worker_id, decision=second_decision)
    assert first.request_identity != second.request_identity
    assert len(intake.submissions) == 2


def test_request_has_no_authority_payload_fields() -> None:
    bridge, worker_repo, responsibility_repo, goal_repo, intake = _bridge()
    worker_id, _, goal_id = _seed_active_chain(
        worker_repo=worker_repo,
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
    )
    bridge.submit_from_decision(
        worker_instance_id=worker_id,
        decision=_action_required_decision(goal_id=goal_id),
    )
    request = intake.submissions[0]
    field_names = {field.name for field in request.__class__.__dataclass_fields__.values()}
    forbidden = {
        "permissions",
        "roles",
        "authority_scopes",
        "admin",
        "requested_authority_scopes",
        "collaborative_authority_scopes",
    }
    assert forbidden.isdisjoint(field_names)


def test_unavailable_adapter_returns_unavailable_without_fake_workitem() -> None:
    worker_repo = InMemoryWorkerInstanceRepository()
    responsibility_repo = InMemoryResponsibilityRepository()
    goal_repo = InMemoryWorkerGoalRepository()
    unavailable = UnavailableCollaborativeWorkIntake()
    bridge = WorkerCollaborativeWorkBridge(
        worker_instance_repository=worker_repo,
        responsibility_repository=responsibility_repo,
        worker_goal_repository=goal_repo,
        intake_port=unavailable,
        clock=lambda: _NOW,
    )
    worker_id, _, goal_id = _seed_active_chain(
        worker_repo=worker_repo,
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
    )
    result = bridge.submit_from_decision(
        worker_instance_id=worker_id,
        decision=_action_required_decision(goal_id=goal_id),
    )
    assert result.disposition == CollaborativeWorkSubmissionDisposition.UNAVAILABLE
    assert result.collaborative_work_ref is None


def test_goal_cancelled_after_decision_rejects_on_reload() -> None:
    bridge, worker_repo, responsibility_repo, goal_repo, intake = _bridge()
    worker_id, _, goal_id = _seed_active_chain(
        worker_repo=worker_repo,
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
    )
    decision = _action_required_decision(goal_id=goal_id)
    stored_goal = goal_repo.get(goal_id=goal_id)
    assert stored_goal is not None
    goal_repo.replace(
        replace(stored_goal, status=WorkerGoalStatus.CANCELLED),
        expected_revision=stored_goal.revision,
    )
    result = bridge.submit_from_decision(worker_instance_id=worker_id, decision=decision)
    assert result.disposition == CollaborativeWorkSubmissionDisposition.REJECTED
    assert intake.submissions == ()


def test_no_action_decision_raises_without_submission() -> None:
    bridge, worker_repo, responsibility_repo, goal_repo, intake = _bridge()
    worker_id, _, goal_id = _seed_active_chain(
        worker_repo=worker_repo,
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
    )
    with pytest.raises(WorkerCollaborativeWorkBridgeRejected):
        bridge.submit_from_decision(
            worker_instance_id=worker_id,
            decision=_no_action_decision(goal_id),
        )
    assert intake.submissions == ()

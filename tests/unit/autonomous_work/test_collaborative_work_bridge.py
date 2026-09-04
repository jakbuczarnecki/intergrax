# © Artur Czarnecki. All rights reserved.

"""AW-4C — collaborative work bridge tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

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
    Revision,
    WorkerGoalStatus,
    WorkerLifecycleState,
    initial_revision,
    mint_wake_up_id,
    mint_worker_goal_id,
    mint_worker_instance_id,
)
from intergrax.contracts.autonomous_work.collaborative_work_bridge import (
    CollaborativeWorkBridgeRejectionReason,
    CollaborativeWorkRequest,
    CollaborativeWorkSubmissionDisposition,
    are_collaborative_work_requests_equivalent,
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
    goal_revision: Revision | None = None,
) -> GoalEvaluationDecision:
    return GoalEvaluationDecision(
        goal_id=goal_id,
        goal_revision=goal_revision or initial_revision(),
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
        goal_revision=initial_revision(),
        evaluated_at=_NOW,
        disposition=GoalEvaluationDisposition.NO_ACTION,
        reason="criteria met",
        wake_up_id=mint_wake_up_id(),
        reason_code=GoalEvaluationReasonCode.CRITERIA_MET,
    )


def _collaborative_work_request(
    *,
    worker_id: str,
    responsibility_id: str,
    goal_id: str,
    wake_up_id: str,
    goal_revision: Revision | None = None,
    reason: str = "SLA risk requires collaborative work",
    evidence_refs: tuple[str, ...] = ("evidence/sla/at-risk",),
    requested_at: datetime | None = None,
) -> CollaborativeWorkRequest:
    request_identity = derive_collaborative_work_request_identity(
        worker_instance_id=worker_id,
        goal_id=goal_id,
        wake_up_id=wake_up_id,
    )
    return CollaborativeWorkRequest(
        request_identity=request_identity,
        worker_instance_id=worker_id,
        responsibility_id=responsibility_id,
        goal_id=goal_id,
        goal_revision=goal_revision or initial_revision(),
        wake_up_id=wake_up_id,
        decision_disposition=GoalEvaluationDisposition.ACTION_REQUIRED,
        reason=reason,
        reason_code=GoalEvaluationReasonCode.SLA_RISK,
        evidence_refs=evidence_refs,
        progress_projection_ref=_PROJECTION_REF,
        requested_priority=contract_suite.worker_goal().priority,
        evaluated_at=_NOW,
        requested_at=requested_at or _NOW,
        title=contract_suite.worker_goal(progress_projection_ref=_PROJECTION_REF).objective,
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
    assert request.goal_revision == decision.goal_revision
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
        goal_revision=initial_revision(),
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
    stored_goal = goal_repo.get(goal_id=goal_id)
    assert stored_goal is not None
    decision = _action_required_decision(
        goal_id=goal_id,
        goal_revision=stored_goal.revision,
    )
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


def test_active_modified_goal_rejects_stale_decision_revision() -> None:
    bridge, worker_repo, responsibility_repo, goal_repo, intake = _bridge()
    worker_id, _, goal_id = _seed_active_chain(
        worker_repo=worker_repo,
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
    )
    stored_goal = goal_repo.get(goal_id=goal_id)
    assert stored_goal is not None
    decision = _action_required_decision(
        goal_id=goal_id,
        goal_revision=stored_goal.revision,
    )
    goal_repo.replace(
        replace(
            stored_goal,
            objective="revised objective while still active",
            priority=stored_goal.priority + 1,
            progress_projection_ref=ProgressProjectionRef("projection/sla-45m"),
        ),
        expected_revision=stored_goal.revision,
    )
    result = bridge.submit_from_decision(worker_instance_id=worker_id, decision=decision)
    assert result.disposition == CollaborativeWorkSubmissionDisposition.REJECTED
    assert result.rejection_reason == CollaborativeWorkBridgeRejectionReason.STALE_OR_NOT_ELIGIBLE
    assert intake.submissions == ()


@pytest.mark.parametrize(
    ("identity_field", "wrong_value"),
    [
        ("worker_instance_id", mint_worker_instance_id()),
        ("goal_id", mint_worker_goal_id()),
        ("wake_up_id", mint_wake_up_id()),
    ],
)
def test_request_identity_must_match_payload_fields(
    identity_field: str,
    wrong_value: str,
) -> None:
    worker_id = contract_suite.worker_instance().worker_instance_id
    goal_id = contract_suite.worker_goal().goal_id
    wake_up_id = mint_wake_up_id()
    identity_kwargs = {
        "worker_instance_id": worker_id,
        "goal_id": goal_id,
        "wake_up_id": wake_up_id,
    }
    identity_kwargs[identity_field] = wrong_value
    request_identity = derive_collaborative_work_request_identity(**identity_kwargs)
    with pytest.raises(ValueError, match="must match"):
        CollaborativeWorkRequest(
            request_identity=request_identity,
            worker_instance_id=worker_id,
            responsibility_id=contract_suite.responsibility().responsibility_id,
            goal_id=goal_id,
            goal_revision=initial_revision(),
            wake_up_id=wake_up_id,
            decision_disposition=GoalEvaluationDisposition.ACTION_REQUIRED,
            reason="SLA risk requires collaborative work",
            reason_code=GoalEvaluationReasonCode.SLA_RISK,
            evidence_refs=("evidence/sla/at-risk",),
            progress_projection_ref=_PROJECTION_REF,
            requested_priority=contract_suite.worker_goal().priority,
            evaluated_at=_NOW,
            requested_at=_NOW,
            title=contract_suite.worker_goal().objective,
        )


def test_conflicting_reason_returns_conflict_without_overwrite() -> None:
    intake = RecordingCollaborativeWorkIntake()
    worker_id = contract_suite.worker_instance().worker_instance_id
    responsibility_id = contract_suite.responsibility().responsibility_id
    goal_id = contract_suite.worker_goal().goal_id
    wake_up_id = mint_wake_up_id()
    first = _collaborative_work_request(
        worker_id=worker_id,
        responsibility_id=responsibility_id,
        goal_id=goal_id,
        wake_up_id=wake_up_id,
    )
    second = _collaborative_work_request(
        worker_id=worker_id,
        responsibility_id=responsibility_id,
        goal_id=goal_id,
        wake_up_id=wake_up_id,
        reason="different collaborative work reason",
    )
    accepted = intake.submit(first)
    conflict = intake.submit(second)
    assert accepted.disposition == CollaborativeWorkSubmissionDisposition.ACCEPTED
    assert conflict.disposition == CollaborativeWorkSubmissionDisposition.CONFLICT
    assert len(intake.submissions) == 1
    assert intake.submissions[0].reason == first.reason


def test_conflicting_evidence_returns_conflict_without_overwrite() -> None:
    intake = RecordingCollaborativeWorkIntake()
    worker_id = contract_suite.worker_instance().worker_instance_id
    responsibility_id = contract_suite.responsibility().responsibility_id
    goal_id = contract_suite.worker_goal().goal_id
    wake_up_id = mint_wake_up_id()
    first = _collaborative_work_request(
        worker_id=worker_id,
        responsibility_id=responsibility_id,
        goal_id=goal_id,
        wake_up_id=wake_up_id,
    )
    second = _collaborative_work_request(
        worker_id=worker_id,
        responsibility_id=responsibility_id,
        goal_id=goal_id,
        wake_up_id=wake_up_id,
        evidence_refs=("evidence/sla/other",),
    )
    intake.submit(first)
    conflict = intake.submit(second)
    assert conflict.disposition == CollaborativeWorkSubmissionDisposition.CONFLICT
    assert intake.submissions[0].evidence_refs == first.evidence_refs


def test_conflicting_goal_revision_returns_conflict_without_overwrite() -> None:
    intake = RecordingCollaborativeWorkIntake()
    worker_id = contract_suite.worker_instance().worker_instance_id
    responsibility_id = contract_suite.responsibility().responsibility_id
    goal_id = contract_suite.worker_goal().goal_id
    wake_up_id = mint_wake_up_id()
    first = _collaborative_work_request(
        worker_id=worker_id,
        responsibility_id=responsibility_id,
        goal_id=goal_id,
        wake_up_id=wake_up_id,
        goal_revision=initial_revision(),
    )
    second = _collaborative_work_request(
        worker_id=worker_id,
        responsibility_id=responsibility_id,
        goal_id=goal_id,
        wake_up_id=wake_up_id,
        goal_revision=Revision(1),
    )
    intake.submit(first)
    conflict = intake.submit(second)
    assert conflict.disposition == CollaborativeWorkSubmissionDisposition.CONFLICT
    assert intake.submissions[0].goal_revision == initial_revision()


def test_retry_with_later_requested_at_is_already_exists() -> None:
    intake = RecordingCollaborativeWorkIntake()
    worker_id = contract_suite.worker_instance().worker_instance_id
    responsibility_id = contract_suite.responsibility().responsibility_id
    goal_id = contract_suite.worker_goal().goal_id
    wake_up_id = mint_wake_up_id()
    first = _collaborative_work_request(
        worker_id=worker_id,
        responsibility_id=responsibility_id,
        goal_id=goal_id,
        wake_up_id=wake_up_id,
        requested_at=_NOW,
    )
    second = _collaborative_work_request(
        worker_id=worker_id,
        responsibility_id=responsibility_id,
        goal_id=goal_id,
        wake_up_id=wake_up_id,
        requested_at=_NOW + timedelta(minutes=5),
    )
    assert are_collaborative_work_requests_equivalent(first, second)
    intake.submit(first)
    replay = intake.submit(second)
    assert replay.disposition == CollaborativeWorkSubmissionDisposition.ALREADY_EXISTS
    assert len(intake.submissions) == 1

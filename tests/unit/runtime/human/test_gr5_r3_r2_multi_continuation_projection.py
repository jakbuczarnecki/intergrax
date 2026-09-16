# © Artur Czarnecki. All rights reserved.

"""GR-5-R3-R2 — multi-continuation projection lifecycle and revision integrity."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    ExecutionHumanVerdict,
    PendingExecutionContinuation,
)
from intergrax.contracts.execution_continuation_projection import (
    ExecutionContinuationProjectionError,
    ExecutionContinuationProjectionStatus,
)
from intergrax.contracts.governed_continuation_correlation import (
    ContinuationReason,
    GovernedContinuationCorrelation,
)
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    mint_attempt_id,
    mint_execution_id,
)
from intergrax.runtime.execution.continuation.composition import (
    wire_task_execution_continuation_projection_sink,
)
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.task.execution_continuation_projection import (
    continuation_projection_allows_replacement,
    execution_continuation_projection_payload_digest,
)
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import HumanApprovalResolution
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests

pytestmark = pytest.mark.unit

_SEED = "gr5-r3-r2"
_TASK = canonical_task_id_for_tests(_SEED)
_RUN = canonical_run_id_for_tests(_SEED)
_ATTEMPT = mint_attempt_id()
_EXECUTION = mint_execution_id()
_C1 = "cont_episode_1"
_C2 = "cont_episode_2"
_PAUSE_C1 = "pause_c1"
_HR_C1 = "hr_c1"
_PAUSE_C2 = "pause_c2"
_HR_C2 = "hr_c2"
_TENANT = "tenant-gr5-r3-r2"


def _identity(
    *,
    attempt: AttemptId | None = None,
    execution: ExecutionId | None = None,
) -> ExecutionContinuationIdentity:
    return ExecutionContinuationIdentity(
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=attempt or _ATTEMPT,
        execution_id=execution or _EXECUTION,
    )


def _governed(
    continuation_id: str,
    *,
    operation_id: str = "op_gr5_r3_r2",
    scope_digest: str | None = None,
) -> GovernedContinuationCorrelation:
    return GovernedContinuationCorrelation(
        continuation_request_id=continuation_id,
        reason=ContinuationReason.SECURITY,
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
        operation_id=operation_id,
        side_effect_scope_digest=scope_digest,
    )


def _task() -> Task:
    return Task(
        task_id=_TASK,
        tenant_id=_TENANT,
        user_id="user-gr5-r3-r2",
        message="gr5-r3-r2",
        run_id=_RUN,
    )


def _pending(
    continuation_id: str,
    state: ExecutionContinuationLifecycleState,
    *,
    revision: int = 1,
    pause_id: str | None = None,
    human_request_id: str | None = None,
    human_verdict: ExecutionHumanVerdict | None = None,
    identity: ExecutionContinuationIdentity | None = None,
    governed: GovernedContinuationCorrelation | None = None,
    requested_at: str = "2026-09-16T10:00:00Z",
) -> PendingExecutionContinuation:
    if state in {
        ExecutionContinuationLifecycleState.PAUSED,
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
        ExecutionContinuationLifecycleState.ESCALATED,
    }:
        pause_id = pause_id or (_PAUSE_C1 if continuation_id == _C1 else _PAUSE_C2)
        human_request_id = human_request_id or (_HR_C1 if continuation_id == _C1 else _HR_C2)
    return PendingExecutionContinuation(
        continuation_id=continuation_id,
        identity=identity or _identity(),
        lifecycle_state=state,
        revision=revision,
        reason=ContinuationReason.SECURITY,
        governed_correlation=governed or _governed(continuation_id),
        pause_id=pause_id,
        human_request_id=human_request_id,
        human_verdict=human_verdict,
        requested_at=requested_at,
    )


def _sink(task: Task):
    return wire_task_execution_continuation_projection_sink(task)


def _gov_snapshot(task: Task) -> dict[str, object]:
    gov = task.runtime.governance
    return {
        "id": gov.projected_continuation_id,
        "revision": gov.projected_continuation_revision,
        "lifecycle": gov.projected_continuation_lifecycle_state,
        "digest": gov.projected_continuation_payload_digest,
        "pause_id": gov.pause_record.pause_id if gov.pause_record else None,
        "hitl": copy.deepcopy(gov.hitl_resolution),
        "grants": (
            gov.governed_continuation_grant,
            gov.physical_delegation_continuation_grant,
        ),
        "verdict": task.options.human.verdict,
        "human_pause": task.options.human.pause_id,
        "human_hr": task.options.human.human_request_id,
    }


def _project_c1_resumed(task: Task) -> None:
    sink = _sink(task)
    for rev, state in (
        (1, ExecutionContinuationLifecycleState.PAUSE_REQUESTED),
        (2, ExecutionContinuationLifecycleState.PAUSED),
        (3, ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN),
        (
            4,
            ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
        ),
        (5, ExecutionContinuationLifecycleState.RESUMED),
    ):
        extra = {}
        if state is ExecutionContinuationLifecycleState.RESUME_AUTHORIZED:
            extra["human_verdict"] = ExecutionHumanVerdict.APPROVE
        if state is ExecutionContinuationLifecycleState.RESUMED:
            extra["human_verdict"] = ExecutionHumanVerdict.APPROVE
        result = sink.project(
            _pending(_C1, state, revision=rev, **extra),
        )
        assert result.status is ExecutionContinuationProjectionStatus.APPLIED


def test_multi_episode_c1_resumed_then_c2_progression() -> None:
    task = _task()
    _project_c1_resumed(task)
    sink = _sink(task)
    for rev, state in (
        (1, ExecutionContinuationLifecycleState.PAUSE_REQUESTED),
        (2, ExecutionContinuationLifecycleState.PAUSED),
        (3, ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN),
    ):
        result = sink.project(_pending(_C2, state, revision=rev))
        assert result.status is ExecutionContinuationProjectionStatus.APPLIED
    assert task.runtime.governance.projected_continuation_id == _C2
    assert task.runtime.governance.projected_continuation_revision == 3


def test_active_c1_blocks_c2_projection() -> None:
    task = _task()
    _sink(task).project(
        _pending(_C1, ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN, revision=3),
    )
    before = _gov_snapshot(task)
    with pytest.raises(ExecutionContinuationProjectionError, match="different continuation_id"):
        HumanPauseCoordinator.project_continuation(
            task,
            _pending(_C2, ExecutionContinuationLifecycleState.PAUSE_REQUESTED, revision=1),
        )
    assert _gov_snapshot(task) == before


def test_resume_authorized_blocks_c2() -> None:
    task = _task()
    _sink(task).project(
        _pending(
            _C1,
            ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
            revision=4,
            human_verdict=ExecutionHumanVerdict.APPROVE,
        ),
    )
    with pytest.raises(ExecutionContinuationProjectionError, match="different continuation_id"):
        _sink(task).project(
            _pending(_C2, ExecutionContinuationLifecycleState.PAUSE_REQUESTED, revision=1),
        )


def test_revision_reset_after_c1_terminal() -> None:
    task = _task()
    _sink(task).project(
        _pending(_C1, ExecutionContinuationLifecycleState.RESUMED, revision=100),
    )
    result = _sink(task).project(
        _pending(_C2, ExecutionContinuationLifecycleState.PAUSE_REQUESTED, revision=1),
    )
    assert result.status is ExecutionContinuationProjectionStatus.APPLIED
    assert task.runtime.governance.projected_continuation_revision == 1


def test_stale_within_c2_after_progression() -> None:
    task = _task()
    _project_c1_resumed(task)
    sink = _sink(task)
    sink.project(_pending(_C2, ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN, revision=3))
    result = sink.project(
        _pending(_C2, ExecutionContinuationLifecycleState.PAUSED, revision=2),
    )
    assert result.status is ExecutionContinuationProjectionStatus.STALE_IGNORED
    assert task.runtime.governance.projected_continuation_revision == 3


def test_same_revision_true_replay_no_drift() -> None:
    task = _task()
    pending = _pending(
        _C1,
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        revision=3,
    )
    sink = _sink(task)
    sink.project(pending)
    snap = _gov_snapshot(task)
    sink.project(pending)
    assert _gov_snapshot(task) == snap


def test_replacement_clears_c1_hitl_and_grants() -> None:
    task = _task()
    sink = _sink(task)
    sink.project(
        _pending(_C1, ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN, revision=3),
    )
    approver = local_development_approver_evidence(actor_id="op", tenant_id=_TENANT)
    gov = task.runtime.governance
    gov.hitl_resolution = HumanApprovalResolution(
        task_id=_TASK,
        pause_id=_PAUSE_C1,
        human_request_id=_HR_C1,
        verdict="approve",
        approver=approver,
        resolved_at="2026-09-16T11:00:00Z",
        run_id=_RUN,
    )
    task.options.human.pause_id = _PAUSE_C1
    sink.project(_pending(_C1, ExecutionContinuationLifecycleState.RESUMED, revision=5))
    assert gov.hitl_resolution is not None
    sink.project(_pending(_C2, ExecutionContinuationLifecycleState.PAUSE_REQUESTED, revision=1))
    assert gov.hitl_resolution is None
    assert task.options.human.pause_id is None


@pytest.mark.parametrize(
    ("field", "update"),
    [
        ("lifecycle_state", {"lifecycle_state": ExecutionContinuationLifecycleState.PAUSED}),
        ("human_verdict", {"human_verdict": ExecutionHumanVerdict.APPROVE}),
        ("pause_id", {"pause_id": "other_pause"}),
        ("human_request_id", {"human_request_id": "other_hr"}),
        ("requested_at", {"requested_at": "2026-09-16T12:00:00Z"}),
        (
            "run_id",
            {
                "identity": replace(
                    _identity(),
                    run_id=canonical_run_id_for_tests("gr5-r3-r2-alt-run"),
                ),
            },
        ),
        (
            "attempt_id",
            {"identity": _identity(attempt=mint_attempt_id(), execution=_EXECUTION)},
        ),
        (
            "execution_id",
            {"identity": _identity(attempt=_ATTEMPT, execution=mint_execution_id())},
        ),
        (
            "governed_digest",
            {
                "governed_correlation": _governed(
                    _C1,
                    scope_digest="sha256:" + ("a" * 64),
                ),
            },
        ),
    ],
)
def test_same_revision_payload_conflict_matrix(
    field: str,
    update: dict[str, object],
) -> None:
    task = _task()
    base = _pending(_C1, ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN, revision=3)
    _sink(task).project(base)
    before = _gov_snapshot(task)
    mutated = base.model_copy(update=update)
    with pytest.raises(
        ExecutionContinuationProjectionError,
        match="revision payload conflict",
    ):
        _sink(task).project(mutated)
    assert _gov_snapshot(task) == before


def test_fingerprint_stable_across_object_rebuild() -> None:
    a = _pending(_C1, ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN, revision=2)
    b = _pending(_C1, ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN, revision=2)
    assert execution_continuation_projection_payload_digest(a) == (
        execution_continuation_projection_payload_digest(b)
    )


def test_fingerprint_differs_on_material_change() -> None:
    a = _pending(_C1, ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN, revision=2)
    b = a.model_copy(update={"pause_id": "other"})
    assert execution_continuation_projection_payload_digest(a) != (
        execution_continuation_projection_payload_digest(b)
    )


@pytest.mark.parametrize(
    ("state", "allows"),
    [
        (ExecutionContinuationLifecycleState.PAUSE_REQUESTED, False),
        (ExecutionContinuationLifecycleState.PAUSED, False),
        (ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN, False),
        (ExecutionContinuationLifecycleState.RESUME_AUTHORIZED, False),
        (ExecutionContinuationLifecycleState.RESUMED, True),
        (ExecutionContinuationLifecycleState.REJECTED, True),
        (ExecutionContinuationLifecycleState.ESCALATED, True),
        (ExecutionContinuationLifecycleState.CANCELLED, True),
    ],
)
def test_replacement_eligibility_semantics(
    state: ExecutionContinuationLifecycleState,
    allows: bool,
) -> None:
    assert continuation_projection_allows_replacement(state.value) is allows


def test_c1_terminal_projection_failure_then_retry_then_c2() -> None:
    task = _task()
    sink = _sink(task)
    sink.project(_pending(_C1, ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN, revision=3))
    resumed = _pending(_C1, ExecutionContinuationLifecycleState.RESUMED, revision=5)
    task.runtime.governance.projected_continuation_lifecycle_state = (
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN.value
    )
    with pytest.raises(ExecutionContinuationProjectionError, match="different continuation_id"):
        sink.project(_pending(_C2, ExecutionContinuationLifecycleState.PAUSE_REQUESTED, revision=1))
    sink.project(resumed)
    result = sink.project(
        _pending(_C2, ExecutionContinuationLifecycleState.PAUSE_REQUESTED, revision=1),
    )
    assert result.status is ExecutionContinuationProjectionStatus.APPLIED

# © Artur Czarnecki. All rights reserved.

"""GR-5-R3-R3 — fail-closed integrity for legacy projections without payload digest."""

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
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    mint_attempt_id,
    mint_execution_id,
)
from intergrax.runtime.execution.continuation.composition import (
    wire_task_execution_continuation_projection_sink,
)
from intergrax.runtime.task.task import Task
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests

pytestmark = pytest.mark.unit

_SEED = "gr5-r3-r3"
_TASK = canonical_task_id_for_tests(_SEED)
_RUN = canonical_run_id_for_tests(_SEED)
_ATTEMPT = mint_attempt_id()
_EXECUTION = mint_execution_id()
_C1 = "cont_legacy_1"
_C2 = "cont_legacy_2"
_PAUSE = "pause_legacy"
_HR = "hr_legacy"
_TENANT = "tenant-gr5-r3-r3"
_LEGACY_UNVERIFIABLE = (
    "legacy projection lacks payload digest; same-revision equality cannot be verified"
)


def _identity(
    *,
    attempt: AttemptId | None = None,
    execution: ExecutionId | None = None,
    run_id=None,
) -> ExecutionContinuationIdentity:
    return ExecutionContinuationIdentity(
        task_id=_TASK,
        run_id=run_id or _RUN,
        attempt_id=attempt or _ATTEMPT,
        execution_id=execution or _EXECUTION,
    )


def _governed(
    continuation_id: str,
    *,
    operation_id: str = "op_gr5_r3_r3",
    scope_digest: str | None = None,
    reason: ContinuationReason = ContinuationReason.SECURITY,
) -> GovernedContinuationCorrelation:
    return GovernedContinuationCorrelation(
        continuation_request_id=continuation_id,
        reason=reason,
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
        user_id="user-gr5-r3-r3",
        message="gr5-r3-r3",
        run_id=_RUN,
    )


def _pending(
    continuation_id: str,
    state: ExecutionContinuationLifecycleState,
    *,
    revision: int = 5,
    pause_id: str | None = _PAUSE,
    human_request_id: str | None = _HR,
    human_verdict: ExecutionHumanVerdict | None = None,
    identity: ExecutionContinuationIdentity | None = None,
    governed: GovernedContinuationCorrelation | None = None,
    requested_at: str = "2026-09-16T10:00:00Z",
    reason: ContinuationReason = ContinuationReason.SECURITY,
) -> PendingExecutionContinuation:
    return PendingExecutionContinuation(
        continuation_id=continuation_id,
        identity=identity or _identity(),
        lifecycle_state=state,
        revision=revision,
        reason=reason,
        governed_correlation=governed or _governed(continuation_id, reason=reason),
        pause_id=pause_id,
        human_request_id=human_request_id,
        human_verdict=human_verdict,
        requested_at=requested_at,
    )


def _sink(task: Task):
    return wire_task_execution_continuation_projection_sink(task)


def _task_snapshot(task: Task) -> dict[str, object]:
    gov = task.runtime.governance
    return {
        "gov": {
            "id": gov.projected_continuation_id,
            "revision": gov.projected_continuation_revision,
            "lifecycle": gov.projected_continuation_lifecycle_state,
            "digest": gov.projected_continuation_payload_digest,
            "paused": gov.paused,
            "pause_id": gov.pause_record.pause_id if gov.pause_record else None,
            "verdict": task.options.human.verdict,
        },
        "metadata": copy.deepcopy(task.metadata),
    }


def _strip_digest(task: Task) -> None:
    task.runtime.governance.projected_continuation_payload_digest = None


def _seed_legacy_same_revision(task: Task, pending: PendingExecutionContinuation) -> None:
    _sink(task).project(pending)
    _strip_digest(task)


def test_legacy_same_revision_no_digest_fail_closed() -> None:
    task = _task()
    pending = _pending(
        _C1,
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        revision=5,
    )
    _seed_legacy_same_revision(task, pending)
    before = _task_snapshot(task)
    with pytest.raises(ExecutionContinuationProjectionError, match=_LEGACY_UNVERIFIABLE):
        _sink(task).project(pending)
    assert _task_snapshot(task) == before


@pytest.mark.parametrize(
    ("field", "update"),
    [
        ("human_verdict", {"human_verdict": ExecutionHumanVerdict.APPROVE}),
        ("reason", {"reason": ContinuationReason.COMPLIANCE}),
        (
            "run_id",
            {
                "identity": replace(
                    _identity(),
                    run_id=canonical_run_id_for_tests("gr5-r3-r3-alt-run"),
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
                    scope_digest="sha256:" + ("b" * 64),
                ),
            },
        ),
        (
            "governed_operation",
            {"governed_correlation": _governed(_C1, operation_id="other_op")},
        ),
        ("requested_at", {"requested_at": "2026-09-16T12:00:00Z"}),
    ],
)
def test_legacy_same_revision_conflict_fail_closed(
    field: str,
    update: dict[str, object],
) -> None:
    del field
    task = _task()
    base = _pending(
        _C1,
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        revision=5,
    )
    _seed_legacy_same_revision(task, base)
    before = _task_snapshot(task)
    mutated = base.model_copy(update=update)
    with pytest.raises(ExecutionContinuationProjectionError):
        _sink(task).project(mutated)
    assert _task_snapshot(task) == before


def test_legacy_newer_revision_installs_digest() -> None:
    task = _task()
    base = _pending(
        _C1,
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        revision=5,
    )
    _seed_legacy_same_revision(task, base)
    newer = _pending(
        _C1,
        ExecutionContinuationLifecycleState.PAUSED,
        revision=6,
    )
    result = _sink(task).project(newer)
    assert result.status is ExecutionContinuationProjectionStatus.APPLIED
    assert task.runtime.governance.projected_continuation_revision == 6
    assert task.runtime.governance.projected_continuation_payload_digest is not None


def test_legacy_stale_revision_stale_ignored() -> None:
    task = _task()
    base = _pending(
        _C1,
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        revision=5,
    )
    _seed_legacy_same_revision(task, base)
    before = _task_snapshot(task)
    result = _sink(task).project(
        _pending(_C1, ExecutionContinuationLifecycleState.PAUSED, revision=4),
    )
    assert result.status is ExecutionContinuationProjectionStatus.STALE_IGNORED
    assert _task_snapshot(task) == before


def test_legacy_terminal_c1_then_c2() -> None:
    task = _task()
    resumed = _pending(
        _C1,
        ExecutionContinuationLifecycleState.RESUMED,
        revision=5,
        human_verdict=ExecutionHumanVerdict.APPROVE,
        pause_id=None,
        human_request_id=None,
    )
    _sink(task).project(resumed)
    _strip_digest(task)
    pending_c2 = _pending(_C2, ExecutionContinuationLifecycleState.PAUSE_REQUESTED, revision=1)
    result = _sink(task).project(pending_c2)
    assert result.status is ExecutionContinuationProjectionStatus.APPLIED
    assert task.runtime.governance.projected_continuation_id == _C2
    assert task.runtime.governance.projected_continuation_payload_digest is not None


def test_legacy_active_c1_blocks_c2() -> None:
    task = _task()
    active = _pending(
        _C1,
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        revision=5,
    )
    _seed_legacy_same_revision(task, active)
    before = _task_snapshot(task)
    with pytest.raises(ExecutionContinuationProjectionError, match="different continuation_id"):
        _sink(task).project(
            _pending(_C2, ExecutionContinuationLifecycleState.PAUSE_REQUESTED, revision=1),
        )
    assert _task_snapshot(task) == before


def test_normal_digest_idempotent_replay_unchanged() -> None:
    task = _task()
    pending = _pending(
        _C1,
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        revision=5,
    )
    sink = _sink(task)
    sink.project(pending)
    snap = _task_snapshot(task)
    sink.project(pending)
    assert _task_snapshot(task) == snap


def test_normal_digest_conflict_fail_closed() -> None:
    task = _task()
    base = _pending(
        _C1,
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        revision=5,
    )
    _sink(task).project(base)
    before = _task_snapshot(task)
    with pytest.raises(
        ExecutionContinuationProjectionError,
        match="revision payload conflict",
    ):
        _sink(task).project(base.model_copy(update={"pause_id": "other_pause"}))
    assert _task_snapshot(task) == before


def test_legacy_same_revision_never_idempotent_without_digest() -> None:
    """Architecture gate: no unverified same-revision no-op when digest is absent."""
    task = _task()
    pending = _pending(
        _C1,
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        revision=5,
    )
    _seed_legacy_same_revision(task, pending)
    with pytest.raises(ExecutionContinuationProjectionError, match=_LEGACY_UNVERIFIABLE):
        result = _sink(task).project(pending)
        assert result.status is ExecutionContinuationProjectionStatus.APPLIED

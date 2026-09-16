# © Artur Czarnecki. All rights reserved.

"""GR-5-R3 — Task / HumanPauseCoordinator canonical continuation projection."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    ExecutionContinuationLookup,
    ExecutionContinuationResolutionCommand,
    ExecutionContinuationResumeCommand,
    ExecutionHumanVerdict,
    ExecutionPauseRequest,
    PendingExecutionContinuation,
)
from intergrax.contracts.execution_continuation_projection import (
    ExecutionContinuationProjectionError,
    ExecutionContinuationProjectionResult,
    ExecutionContinuationProjectionSink,
    ExecutionContinuationProjectionStatus,
)
from intergrax.contracts.governed_continuation_correlation import (
    ContinuationReason,
    GovernedContinuationCorrelation,
)
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id
from intergrax.runtime.execution.continuation.composition import (
    wire_execution_engine_continuation_dependencies,
    wire_task_execution_continuation_projection_sink,
)
from intergrax.runtime.execution.continuation.progress_gate import (
    assert_canonical_execution_may_progress,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.task.execution_continuation_projection import (
    TaskExecutionContinuationProjectionSink,
    governance_paused_for_lifecycle,
)
from intergrax.runtime.task.task import Task
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests

pytestmark = pytest.mark.unit

_SEED = "gr5-r3-projection"
_TASK = canonical_task_id_for_tests(_SEED)
_RUN = canonical_run_id_for_tests(_SEED)
_ATTEMPT = mint_attempt_id()
_EXECUTION = mint_execution_id()
_CONTINUATION_ID = "gcr_gr5_r3"
_PAUSE = "pause_p1"
_HR = "hr_h1"
_TENANT = "tenant-gr5-r3"


def _identity() -> ExecutionContinuationIdentity:
    return ExecutionContinuationIdentity(
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
    )


def _governed() -> GovernedContinuationCorrelation:
    return GovernedContinuationCorrelation(
        continuation_request_id=_CONTINUATION_ID,
        reason=ContinuationReason.SECURITY,
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
        operation_id="op_gr5_r3",
    )


def _task() -> Task:
    return Task(
        task_id=_TASK,
        tenant_id=_TENANT,
        user_id="user-gr5-r3",
        message="gr5-r3",
        run_id=_RUN,
    )


def _pending(
    state: ExecutionContinuationLifecycleState,
    *,
    revision: int = 1,
    human_verdict: ExecutionHumanVerdict | None = None,
) -> PendingExecutionContinuation:
    return PendingExecutionContinuation(
        continuation_id=_CONTINUATION_ID,
        identity=_identity(),
        lifecycle_state=state,
        revision=revision,
        reason=ContinuationReason.SECURITY,
        governed_correlation=_governed(),
        pause_id=_PAUSE,
        human_request_id=_HR,
        human_verdict=human_verdict,
        requested_at="2026-09-15T10:00:00Z",
    )


def _project(task: Task, pending: PendingExecutionContinuation) -> None:
    HumanPauseCoordinator.project_continuation(task, pending)


@pytest.mark.parametrize(
    ("state", "expect_paused", "expect_resumed"),
    [
        (ExecutionContinuationLifecycleState.PAUSE_REQUESTED, False, False),
        (ExecutionContinuationLifecycleState.PAUSED, True, False),
        (ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN, True, False),
        (ExecutionContinuationLifecycleState.RESUME_AUTHORIZED, True, False),
        (ExecutionContinuationLifecycleState.RESUMED, False, True),
        (ExecutionContinuationLifecycleState.REJECTED, False, False),
        (ExecutionContinuationLifecycleState.ESCALATED, True, False),
        (ExecutionContinuationLifecycleState.CANCELLED, False, False),
    ],
)
def test_lifecycle_projection_paused_and_resumed_flags(
    state: ExecutionContinuationLifecycleState,
    expect_paused: bool,
    expect_resumed: bool,
) -> None:
    task = _task()
    _project(task, _pending(state))
    assert task.runtime.governance.paused is expect_paused
    assert HumanPauseCoordinator.is_resumed(task) is expect_resumed
    assert governance_paused_for_lifecycle(state) is expect_paused


def test_resume_authorized_does_not_show_fully_resumed() -> None:
    task = _task()
    _project(
        task,
        _pending(
            ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
            human_verdict=ExecutionHumanVerdict.APPROVE,
        ),
    )
    assert HumanPauseCoordinator.is_resumed(task) is False
    assert task.options.human.verdict == "approve"


def test_exact_pause_and_human_request_ids() -> None:
    task = _task()
    _project(task, _pending(ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN))
    record = task.runtime.governance.pause_record
    assert record is not None
    assert record.pause_id == _PAUSE
    assert record.human_request_id == _HR


def test_four_id_binding_on_pause_record() -> None:
    task = _task()
    _project(task, _pending(ExecutionContinuationLifecycleState.PAUSED))
    record = task.runtime.governance.pause_record
    assert record is not None
    assert record.task_id == _TASK


def test_stale_revision_ignored() -> None:
    task = _task()
    sink = wire_task_execution_continuation_projection_sink(task)
    sink.project(_pending(ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN, revision=5))
    result = sink.project(
        _pending(ExecutionContinuationLifecycleState.RESUMED, revision=4),
    )
    assert result.status is ExecutionContinuationProjectionStatus.STALE_IGNORED
    assert task.runtime.governance.projected_continuation_revision == 5


def test_idempotent_replay_same_revision() -> None:
    task = _task()
    pending = _pending(ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN, revision=3)
    sink = wire_task_execution_continuation_projection_sink(task)
    first = sink.project(pending)
    second = sink.project(pending)
    assert first.status is ExecutionContinuationProjectionStatus.APPLIED
    assert second.status is ExecutionContinuationProjectionStatus.APPLIED
    assert task.runtime.governance.pause_record is not None
    assert task.runtime.governance.pause_record.pause_id == _PAUSE


@dataclass
class _FailingSink:
    def project(
        self,
        pending: PendingExecutionContinuation,
    ) -> ExecutionContinuationProjectionResult:
        raise ExecutionContinuationProjectionError("sink failed")


def test_projection_failure_does_not_mutate_canonical() -> None:
    deps = wire_execution_engine_continuation_dependencies()
    port = deps.continuation
    pending = port.request_pause(
        ExecutionPauseRequest(
            identity=_identity(),
            continuation_id=_CONTINUATION_ID,
            reason=ContinuationReason.SECURITY,
            pause_id=_PAUSE,
            human_request_id=_HR,
            governed_correlation=_governed(),
        ),
    )
    task = _task()
    with pytest.raises(ExecutionContinuationProjectionError):
        HumanPauseCoordinator.project_continuation(task, pending, projection_sink=_FailingSink())
    reloaded = port.get_pending(ExecutionContinuationLookup(continuation_id=_CONTINUATION_ID))
    assert reloaded.lifecycle_state is pending.lifecycle_state
    assert reloaded.revision == pending.revision


class _AltSink(TaskExecutionContinuationProjectionSink):
    """Second implementation proving contract replaceability."""

    def project(
        self,
        pending: PendingExecutionContinuation,
    ) -> ExecutionContinuationProjectionResult:
        result = super().project(pending)
        self._task.metadata["gr5_r3_alt_projection"] = pending.revision
        return result


def test_custom_projection_implementation() -> None:
    task = _task()
    sink = _AltSink(task)
    sink.project(_pending(ExecutionContinuationLifecycleState.PAUSED))
    assert task.metadata["gr5_r3_alt_projection"] == 1


def test_task_only_human_response_does_not_resume_canonical() -> None:
    deps = wire_execution_engine_continuation_dependencies()
    driver = deps.lifecycle_driver
    port = deps.continuation
    port.request_pause(
        ExecutionPauseRequest(
            identity=_identity(),
            continuation_id=_CONTINUATION_ID,
            reason=ContinuationReason.SECURITY,
            pause_id=_PAUSE,
            human_request_id=_HR,
            governed_correlation=_governed(),
        ),
    )
    driver.record_execution_reached_safe_pause(
        _CONTINUATION_ID,
        execution_pause_established=True,
    )
    waiting = driver.record_ready_for_human_resolution(_CONTINUATION_ID)
    task = _task()
    _project(task, waiting)
    approver = local_development_approver_evidence(actor_id="op", tenant_id=_TENANT)
    HumanPauseCoordinator.resolve_human_response(
        task,
        HumanResponseVerdict.APPROVE,
        approver=approver,
        pause_id=_PAUSE,
        human_request_id=_HR,
        attempt_id=str(_ATTEMPT),
        execution_id=str(_EXECUTION),
    )
    still_waiting = port.get_pending(ExecutionContinuationLookup(continuation_id=_CONTINUATION_ID))
    assert still_waiting.lifecycle_state is ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN


def test_clear_pause_does_not_establish_canonical_resume() -> None:
    deps = wire_execution_engine_continuation_dependencies()
    driver = deps.lifecycle_driver
    port = deps.continuation
    port.request_pause(
        ExecutionPauseRequest(
            identity=_identity(),
            continuation_id=_CONTINUATION_ID,
            reason=ContinuationReason.SECURITY,
            pause_id=_PAUSE,
            human_request_id=_HR,
        ),
    )
    driver.record_execution_reached_safe_pause(
        _CONTINUATION_ID,
        execution_pause_established=True,
    )
    waiting = driver.record_ready_for_human_resolution(_CONTINUATION_ID)
    task = _task()
    _project(task, waiting)
    HumanPauseCoordinator.clear_pause(task)
    assert task.runtime.governance.paused is False
    reloaded = port.get_pending(ExecutionContinuationLookup(continuation_id=_CONTINUATION_ID))
    assert reloaded.lifecycle_state is ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN


def test_task_corruption_does_not_bypass_canonical_gate() -> None:
    deps = wire_execution_engine_continuation_dependencies()
    store = deps.continuation_service.store
    driver = deps.lifecycle_driver
    port = deps.continuation
    port.request_pause(
        ExecutionPauseRequest(
            identity=_identity(),
            continuation_id=_CONTINUATION_ID,
            reason=ContinuationReason.SECURITY,
            pause_id=_PAUSE,
            human_request_id=_HR,
        ),
    )
    driver.record_execution_reached_safe_pause(
        _CONTINUATION_ID,
        execution_pause_established=True,
    )
    waiting = driver.record_ready_for_human_resolution(_CONTINUATION_ID)
    task = _task()
    _project(task, waiting)
    task.runtime.governance.paused = False
    with pytest.raises(ExecutionContinuationError):
        assert_canonical_execution_may_progress(store=store, identity=_identity())


def test_canonical_resumed_repairs_task_projection() -> None:
    deps = wire_execution_engine_continuation_dependencies()
    port = deps.continuation
    driver = deps.lifecycle_driver
    port.request_pause(
        ExecutionPauseRequest(
            identity=_identity(),
            continuation_id=_CONTINUATION_ID,
            reason=ContinuationReason.SECURITY,
            pause_id=_PAUSE,
            human_request_id=_HR,
            governed_correlation=_governed(),
        ),
    )
    driver.record_execution_reached_safe_pause(
        _CONTINUATION_ID,
        execution_pause_established=True,
    )
    waiting = driver.record_ready_for_human_resolution(_CONTINUATION_ID)
    task = _task()
    _project(task, waiting)
    task.runtime.governance.paused = True
    approved = port.apply_resolution(
        ExecutionContinuationResolutionCommand(
            continuation_id=_CONTINUATION_ID,
            identity=_identity(),
            expected_revision=waiting.revision,
            verdict=ExecutionHumanVerdict.APPROVE,
            approver=local_development_approver_evidence(actor_id="op", tenant_id=_TENANT),
            human_request_id=_HR,
            pause_id=_PAUSE,
            operation_id="op_gr5_r3",
            resolved_at="2026-09-15T11:00:00Z",
        ),
    )
    resumed = port.resume(
        ExecutionContinuationResumeCommand(
            continuation_id=_CONTINUATION_ID,
            identity=_identity(),
            expected_revision=approved.revision,
        ),
    )
    _project(task, resumed)
    assert HumanPauseCoordinator.is_resumed(task) is True
    assert task.runtime.governance.paused is False


def test_resolve_human_response_and_apply_canonical_path() -> None:
    deps = wire_execution_engine_continuation_dependencies()
    port = deps.continuation
    driver = deps.lifecycle_driver
    port.request_pause(
        ExecutionPauseRequest(
            identity=_identity(),
            continuation_id=_CONTINUATION_ID,
            reason=ContinuationReason.SECURITY,
            pause_id=_PAUSE,
            human_request_id=_HR,
            governed_correlation=_governed(),
        ),
    )
    driver.record_execution_reached_safe_pause(
        _CONTINUATION_ID,
        execution_pause_established=True,
    )
    waiting = driver.record_ready_for_human_resolution(_CONTINUATION_ID)
    task = _task()
    _project(task, waiting)
    approver = local_development_approver_evidence(actor_id="op", tenant_id=_TENANT)
    updated = HumanPauseCoordinator.resolve_human_response_and_apply_canonical(
        task,
        HumanResponseVerdict.APPROVE,
        approver=approver,
        continuation=port,
        pause_id=_PAUSE,
        human_request_id=_HR,
        attempt_id=str(_ATTEMPT),
        execution_id=str(_EXECUTION),
    )
    assert updated.lifecycle_state is ExecutionContinuationLifecycleState.RESUME_AUTHORIZED
    assert HumanPauseCoordinator.is_resumed(task) is False


def test_falsey_custom_sink_preserved() -> None:
    """Explicit ``is not None`` wiring — falsy sink objects are not replaced."""

    class _ZeroSink:
        def project(
            self,
            pending: PendingExecutionContinuation,
        ) -> ExecutionContinuationProjectionResult:
            return ExecutionContinuationProjectionResult(
                status=ExecutionContinuationProjectionStatus.APPLIED,
                applied_revision=pending.revision,
            )

    task = _task()
    sink: ExecutionContinuationProjectionSink = _ZeroSink()  # type: ignore[assignment]
    result = HumanPauseCoordinator.project_continuation(
        task,
        _pending(ExecutionContinuationLifecycleState.PAUSED),
        projection_sink=sink,
    )
    assert result.applied_revision == 1
    assert task.runtime.governance.paused is False

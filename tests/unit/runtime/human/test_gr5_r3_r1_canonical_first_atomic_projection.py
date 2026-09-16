# © Artur Czarnecki. All rights reserved.

"""GR-5-R3-R1 — canonical-first resolution and atomic Task projection."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import pytest

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    ExecutionContinuationLookup,
    ExecutionContinuationPort,
    ExecutionContinuationResolutionCommand,
    ExecutionHumanVerdict,
    ExecutionPauseRequest,
    PendingExecutionContinuation,
)
from intergrax.contracts.execution_continuation_projection import (
    ExecutionContinuationCanonicalProjectionApplyError,
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
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.task.execution_continuation_projection import (
    prepare_task_continuation_projection,
)
from intergrax.runtime.task.task import Task
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests

pytestmark = pytest.mark.unit

_SEED = "gr5-r3-r1"
_TASK = canonical_task_id_for_tests(_SEED)
_RUN = canonical_run_id_for_tests(_SEED)
_ATTEMPT = mint_attempt_id()
_EXECUTION = mint_execution_id()
_CONTINUATION_ID = "gcr_gr5_r3_r1"
_PAUSE = "pause_r3r1"
_HR = "hr_r3r1"
_TENANT = "tenant-gr5-r3-r1"


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
        operation_id="op_gr5_r3_r1",
    )


def _task() -> Task:
    return Task(
        task_id=_TASK,
        tenant_id=_TENANT,
        user_id="user-gr5-r3-r1",
        message="gr5-r3-r1",
        run_id=_RUN,
    )


def _pending(
    state: ExecutionContinuationLifecycleState,
    *,
    revision: int = 1,
    pause_id: str | None = _PAUSE,
    human_request_id: str | None = _HR,
) -> PendingExecutionContinuation:
    return PendingExecutionContinuation(
        continuation_id=_CONTINUATION_ID,
        identity=_identity(),
        lifecycle_state=state,
        revision=revision,
        reason=ContinuationReason.SECURITY,
        governed_correlation=_governed(),
        pause_id=pause_id,
        human_request_id=human_request_id,
        requested_at="2026-09-16T10:00:00Z",
    )


def _projection_snapshot(task: Task) -> dict[str, object]:
    gov = task.runtime.governance
    return {
        "hitl": copy.deepcopy(gov.hitl_resolution),
        "verdict": task.options.human.verdict,
        "revision": gov.projected_continuation_revision,
        "lifecycle": gov.projected_continuation_lifecycle_state,
        "paused": gov.paused,
        "pause_id": gov.pause_record.pause_id if gov.pause_record else None,
    }


def _waiting_setup() -> tuple[Task, object, PendingExecutionContinuation]:
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
    HumanPauseCoordinator.project_continuation(task, waiting)
    return task, port, waiting


@dataclass
class _RecordingContinuationPort:
    inner: ExecutionContinuationPort
    events: list[str] = field(default_factory=list)

    def get_pending(self, lookup: ExecutionContinuationLookup) -> PendingExecutionContinuation:
        self.events.append("get_pending")
        return self.inner.get_pending(lookup)

    def apply_resolution(
        self,
        command: ExecutionContinuationResolutionCommand,
    ) -> PendingExecutionContinuation:
        self.events.append("apply_resolution")
        return self.inner.apply_resolution(command)

    def request_pause(self, request: ExecutionPauseRequest) -> PendingExecutionContinuation:
        return self.inner.request_pause(request)

    def resume(self, command):  # noqa: ANN001
        return self.inner.resume(command)


@dataclass
class _RecordingSink:
    events: list[str] = field(default_factory=list)

    def project(
        self,
        pending: PendingExecutionContinuation,
    ) -> ExecutionContinuationProjectionResult:
        self.events.append("project")
        return ExecutionContinuationProjectionResult(
            status=ExecutionContinuationProjectionStatus.APPLIED,
            applied_revision=pending.revision,
        )


def test_canonical_success_call_order() -> None:
    task, port, _waiting = _waiting_setup()
    recording_port = _RecordingContinuationPort(port)
    sink = _RecordingSink()
    approver = local_development_approver_evidence(actor_id="op", tenant_id=_TENANT)
    HumanPauseCoordinator.resolve_human_response_and_apply_canonical(
        task,
        HumanResponseVerdict.APPROVE,
        approver=approver,
        continuation=recording_port,
        projection_sink=sink,
        pause_id=_PAUSE,
        human_request_id=_HR,
        attempt_id=str(_ATTEMPT),
        execution_id=str(_EXECUTION),
    )
    assert recording_port.events == ["get_pending", "apply_resolution"]
    assert sink.events == ["project"]


def test_canonical_stale_revision_leaves_task_unchanged() -> None:
    task, port, waiting = _waiting_setup()
    before = _projection_snapshot(task)
    approver = local_development_approver_evidence(actor_id="op", tenant_id=_TENANT)

    class _StalePort:
        def get_pending(self, lookup: ExecutionContinuationLookup) -> PendingExecutionContinuation:
            return port.get_pending(lookup)

        def apply_resolution(
            self,
            command: ExecutionContinuationResolutionCommand,
        ) -> PendingExecutionContinuation:
            raise ExecutionContinuationError(
                "stale",
                code=ExecutionContinuationErrorCode.STALE_REVISION,
            )

        def request_pause(self, request: ExecutionPauseRequest) -> PendingExecutionContinuation:
            return port.request_pause(request)

        def resume(self, command):  # noqa: ANN001
            return port.resume(command)

    with pytest.raises(ExecutionContinuationError) as exc:
        HumanPauseCoordinator.resolve_human_response_and_apply_canonical(
            task,
            HumanResponseVerdict.APPROVE,
            approver=approver,
            continuation=_StalePort(),
            pause_id=_PAUSE,
            human_request_id=_HR,
            attempt_id=str(_ATTEMPT),
            execution_id=str(_EXECUTION),
        )
    assert exc.value.code is ExecutionContinuationErrorCode.STALE_REVISION
    assert _projection_snapshot(task) == before
    reloaded = port.get_pending(ExecutionContinuationLookup(continuation_id=_CONTINUATION_ID))
    assert reloaded.revision == waiting.revision


@pytest.mark.parametrize(
    "code",
    [
        ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
        ExecutionContinuationErrorCode.SCOPE_MISMATCH,
        ExecutionContinuationErrorCode.ALREADY_RESOLVED,
    ],
)
def test_canonical_failure_codes_leave_task_unchanged(code: ExecutionContinuationErrorCode) -> None:
    task, port, _waiting = _waiting_setup()
    before = _projection_snapshot(task)
    approver = local_development_approver_evidence(actor_id="op", tenant_id=_TENANT)

    class _FailPort:
        def get_pending(self, lookup: ExecutionContinuationLookup) -> PendingExecutionContinuation:
            return port.get_pending(lookup)

        def apply_resolution(
            self,
            command: ExecutionContinuationResolutionCommand,
        ) -> PendingExecutionContinuation:
            raise ExecutionContinuationError("fail", code=code)

        def request_pause(self, request: ExecutionPauseRequest) -> PendingExecutionContinuation:
            return port.request_pause(request)

        def resume(self, command):  # noqa: ANN001
            return port.resume(command)

    with pytest.raises(ExecutionContinuationError) as exc:
        HumanPauseCoordinator.resolve_human_response_and_apply_canonical(
            task,
            HumanResponseVerdict.APPROVE,
            approver=approver,
            continuation=_FailPort(),
            pause_id=_PAUSE,
            human_request_id=_HR,
            attempt_id=str(_ATTEMPT),
            execution_id=str(_EXECUTION),
        )
    assert exc.value.code is code
    assert _projection_snapshot(task) == before


def test_projection_validation_failure_leaves_task_unchanged() -> None:
    task = _task()
    HumanPauseCoordinator.project_continuation(
        task,
        _pending(ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN, revision=2),
    )
    before = _projection_snapshot(task)
    invalid = _pending(
        ExecutionContinuationLifecycleState.PAUSED,
        revision=3,
        pause_id=None,
        human_request_id=None,
    )
    with pytest.raises(ExecutionContinuationProjectionError):
        HumanPauseCoordinator.project_continuation(task, invalid)
    assert _projection_snapshot(task) == before


def test_prepare_does_not_mutate_task() -> None:
    task = _task()
    before = _projection_snapshot(task)
    prepare_task_continuation_projection(
        task,
        _pending(ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN),
    )
    assert _projection_snapshot(task) == before


def test_projection_failure_after_canonical_success() -> None:
    task, port, _waiting = _waiting_setup()
    before = _projection_snapshot(task)
    approver = local_development_approver_evidence(actor_id="op", tenant_id=_TENANT)

    class _FailingSink:
        def project(
            self,
            pending: PendingExecutionContinuation,
        ) -> ExecutionContinuationProjectionResult:
            raise ExecutionContinuationProjectionError("projection failed")

    with pytest.raises(ExecutionContinuationCanonicalProjectionApplyError) as exc:
        HumanPauseCoordinator.resolve_human_response_and_apply_canonical(
            task,
            HumanResponseVerdict.APPROVE,
            approver=approver,
            continuation=port,
            projection_sink=_FailingSink(),
            pause_id=_PAUSE,
            human_request_id=_HR,
            attempt_id=str(_ATTEMPT),
            execution_id=str(_EXECUTION),
        )
    assert exc.value.canonical_snapshot.lifecycle_state is (
        ExecutionContinuationLifecycleState.RESUME_AUTHORIZED
    )
    assert _projection_snapshot(task) == before
    reloaded = port.get_pending(ExecutionContinuationLookup(continuation_id=_CONTINUATION_ID))
    assert reloaded.lifecycle_state is ExecutionContinuationLifecycleState.RESUME_AUTHORIZED


def test_reprojection_after_projection_failure() -> None:
    task, port, _waiting = _waiting_setup()
    approver = local_development_approver_evidence(actor_id="op", tenant_id=_TENANT)

    class _FailingSink:
        def project(
            self,
            pending: PendingExecutionContinuation,
        ) -> ExecutionContinuationProjectionResult:
            raise ExecutionContinuationProjectionError("projection failed")

    with pytest.raises(ExecutionContinuationCanonicalProjectionApplyError):
        HumanPauseCoordinator.resolve_human_response_and_apply_canonical(
            task,
            HumanResponseVerdict.APPROVE,
            approver=approver,
            continuation=port,
            projection_sink=_FailingSink(),
            pause_id=_PAUSE,
            human_request_id=_HR,
            attempt_id=str(_ATTEMPT),
            execution_id=str(_EXECUTION),
        )
    updated = port.get_pending(ExecutionContinuationLookup(continuation_id=_CONTINUATION_ID))
    accepted = HumanPauseCoordinator._build_human_approval_resolution(
        task,
        HumanResponseVerdict.APPROVE,
        approver=approver,
        pause_id=_PAUSE,
        human_request_id=_HR,
        run_id=None,
        attempt_id=str(_ATTEMPT),
        execution_id=str(_EXECUTION),
        response_text=None,
    )
    HumanPauseCoordinator.project_continuation(
        task,
        updated,
        accepted_hitl_resolution=accepted,
    )
    assert task.runtime.governance.projected_continuation_lifecycle_state == (
        ExecutionContinuationLifecycleState.RESUME_AUTHORIZED.value
    )
    assert task.runtime.governance.hitl_resolution is not None


def test_same_revision_replay_idempotent() -> None:
    task = _task()
    sink = wire_task_execution_continuation_projection_sink(task)
    pending = _pending(ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN, revision=4)
    first = sink.project(pending)
    snapshot = _projection_snapshot(task)
    second = sink.project(pending)
    assert first.status is ExecutionContinuationProjectionStatus.APPLIED
    assert second.status is ExecutionContinuationProjectionStatus.APPLIED
    assert _projection_snapshot(task) == snapshot


def test_different_continuation_id_projection_rejected() -> None:
    task = _task()
    HumanPauseCoordinator.project_continuation(
        task,
        _pending(ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN),
    )
    other = _pending(ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN).model_copy(
        update={"continuation_id": "other_continuation"},
    )
    with pytest.raises(ExecutionContinuationProjectionError, match="different continuation_id"):
        HumanPauseCoordinator.project_continuation(task, other)


def test_canonical_command_uses_pending_pause_ids() -> None:
    task, port, _waiting = _waiting_setup()
    captured: list[ExecutionContinuationResolutionCommand] = []

    class _CapturePort(_RecordingContinuationPort):
        def apply_resolution(
            self,
            command: ExecutionContinuationResolutionCommand,
        ) -> PendingExecutionContinuation:
            captured.append(command)
            return super().apply_resolution(command)

    recording = _CapturePort(port)
    approver = local_development_approver_evidence(actor_id="op", tenant_id=_TENANT)
    HumanPauseCoordinator.resolve_human_response_and_apply_canonical(
        task,
        HumanResponseVerdict.APPROVE,
        approver=approver,
        continuation=recording,
        pause_id=_PAUSE,
        human_request_id=_HR,
        attempt_id=str(_ATTEMPT),
        execution_id=str(_EXECUTION),
    )
    assert len(captured) == 1
    assert captured[0].pause_id == _PAUSE
    assert captured[0].human_request_id == _HR
    assert captured[0].operation_id == "op_gr5_r3_r1"

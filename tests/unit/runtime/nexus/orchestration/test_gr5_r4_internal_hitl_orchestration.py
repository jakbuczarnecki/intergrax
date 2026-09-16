# © Artur Czarnecki. All rights reserved.

"""GR-5-R4 — internal orchestration canonical HITL alignment."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.contracts.agent_decision import HumanRequest
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationLifecycleState,
    ExecutionContinuationLookup,
    ExecutionContinuationResumeCommand,
    ExecutionContinuationResolutionCommand,
    ExecutionHumanVerdict,
    ExecutionPauseRequest,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.contracts.governed_continuation_correlation import (
    ContinuationReason,
    GovernedContinuationCorrelation,
)
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.execution.continuation.composition import (
    wire_execution_engine_continuation_dependencies,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.nexus.orchestration.internal_continuation_orchestration import (
    InternalOrchestrationContinuation,
    canonical_allows_planning_progress_after_human_gate,
    establish_canonical_hitl_pause,
    execution_continuation_identity_for_task,
)
from intergrax.runtime.nexus.orchestration.intake_runner import NexusIntakeRunner
from intergrax.runtime.nexus.orchestration.planning_runner import NexusPlanningRunner
from intergrax.runtime.nexus.task_classifier import TaskClassification
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_contract import TaskPauseRecord
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from intergrax.runtime.task.task_trace import TaskTraceEmitter
from testing_support.builder import canonical_task_id_for_tests

pytestmark = pytest.mark.unit

_TASK = canonical_task_id_for_tests("gr5-r4")
_RUN = mint_run_id()
_ATTEMPT = mint_attempt_id()
_EXECUTION = mint_execution_id()
_CONTINUATION = "gcr_gr5_r4"
_PAUSE = "pause_gr5_r4"
_HR = "hr_gr5_r4"
_OPERATION = "op_gr5_r4"


def _governed() -> GovernedContinuationCorrelation:
    return GovernedContinuationCorrelation(
        continuation_request_id=_CONTINUATION,
        reason=ContinuationReason.SECURITY,
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
        operation_id=_OPERATION,
    )


def _identity():
    return execution_continuation_identity_for_task(
        Task(tenant_id="t1", user_id="u1", message="m", task_id=_TASK),
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
    )


def _capability() -> InternalOrchestrationContinuation:
    deps = wire_execution_engine_continuation_dependencies()
    return InternalOrchestrationContinuation(
        port=deps.continuation,
        lifecycle_driver=deps.lifecycle_driver,
    )


def _waiting_task(cap: InternalOrchestrationContinuation) -> Task:
    task = Task(tenant_id="t1", user_id="u1", message="m", task_id=_TASK)
    task.runtime.governance.human_request = HumanRequest(
        request_id=_HR,
        prompt="approve?",
        governed_continuation=_governed(),
    )
    establish_canonical_hitl_pause(
        task,
        identity=_identity(),
        continuation_id=_CONTINUATION,
        reason=ContinuationReason.SECURITY,
        pause_id=_PAUSE,
        human_request_id=_HR,
        capability=cap,
        governed_correlation=_governed(),
    )
    return task


def _seed_waiting() -> tuple[InternalOrchestrationContinuation, Task]:
    cap = _capability()
    return cap, _waiting_task(cap)


@pytest.fixture
def bound_identity():
    token = bind_active_execution_identity(
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
        task_id=_TASK,
    )
    yield
    reset_active_execution_identity(token)


def test_task_projection_corruption_cannot_resume_planning(bound_identity: None) -> None:
    cap, task = _seed_waiting()
    task.runtime.governance.paused = False
    task.runtime.governance.hitl_resolution = None
    task.options.human.verdict = "approve"
    identity = _identity()
    assert canonical_allows_planning_progress_after_human_gate(
        cap,
        identity=identity,
        human_approval_required=True,
    ) is False


def test_resume_authorized_blocks_planning(bound_identity: None) -> None:
    cap, task = _seed_waiting()
    pending = cap.port.get_pending(ExecutionContinuationLookup(continuation_id=_CONTINUATION))
    approved = cap.port.apply_resolution(
        ExecutionContinuationResolutionCommand(
            continuation_id=_CONTINUATION,
            identity=_identity(),
            expected_revision=pending.revision,
            verdict=ExecutionHumanVerdict.APPROVE,
            approver=local_development_approver_evidence(actor_id="op", tenant_id="t1"),
            human_request_id=_HR,
            pause_id=_PAUSE,
            operation_id=_OPERATION,
            resolved_at="2026-09-16T10:00:00Z",
        ),
    )
    assert approved.lifecycle_state is ExecutionContinuationLifecycleState.RESUME_AUTHORIZED
    HumanPauseCoordinator.project_continuation(task, approved)
    assert canonical_allows_planning_progress_after_human_gate(
        cap,
        identity=_identity(),
        human_approval_required=True,
    ) is False


def test_resumed_allows_planning_once(bound_identity: None) -> None:
    cap, task = _seed_waiting()
    pending = cap.port.get_pending(ExecutionContinuationLookup(continuation_id=_CONTINUATION))
    approved = cap.port.apply_resolution(
        ExecutionContinuationResolutionCommand(
            continuation_id=_CONTINUATION,
            identity=_identity(),
            expected_revision=pending.revision,
            verdict=ExecutionHumanVerdict.APPROVE,
            approver=local_development_approver_evidence(actor_id="op", tenant_id="t1"),
            human_request_id=_HR,
            pause_id=_PAUSE,
            operation_id=_OPERATION,
            resolved_at="2026-09-16T10:00:00Z",
        ),
    )
    resumed = cap.port.resume(
        ExecutionContinuationResumeCommand(
            continuation_id=_CONTINUATION,
            identity=_identity(),
            expected_revision=approved.revision,
        ),
    )
    HumanPauseCoordinator.project_continuation(task, resumed)
    assert canonical_allows_planning_progress_after_human_gate(
        cap,
        identity=_identity(),
        human_approval_required=True,
    ) is True


@pytest.mark.asyncio
async def test_intake_canonical_resume_owner(bound_identity: None) -> None:
    cap, task = _seed_waiting()
    task.runtime.governance.pause_record = TaskPauseRecord(
        pause_id=_PAUSE,
        task_id=_TASK,
        human_request_id=_HR,
    )
    task.options.human.verdict = HumanResponseVerdict.APPROVE.value
    task.options.human.pause_id = _PAUSE
    task.options.human.human_request_id = _HR
    task.options.human.approver = local_development_approver_evidence(actor_id="op", tenant_id="t1")
    from intergrax.contracts.execution_identity import ActiveExecutionIdentity

    human_hooks = MagicMock()
    human_hooks.after_response = AsyncMock()
    runner = NexusIntakeRunner(
        hitl=MagicMock(),
        human_hooks=human_hooks,
        publish=AsyncMock(),
        restore_long_running=AsyncMock(),
        execution_identity=ActiveExecutionIdentity(),
        hitl_continuation=cap,
    )
    await runner.run(
        task,
        lifecycle=TaskLifecycle(),
        trace_emitter=TaskTraceEmitter(run_id=_RUN, attempt_id=_ATTEMPT),
    )
    snapshot = cap.port.get_pending(ExecutionContinuationLookup(continuation_id=_CONTINUATION))
    assert snapshot.lifecycle_state is ExecutionContinuationLifecycleState.RESUMED


def test_hitl_without_continuation_capability_fails() -> None:
    from intergrax.runtime.nexus.orchestration.internal_continuation_orchestration import (
        InternalHitlContinuationCapabilityError,
        require_internal_hitl_continuation,
    )

    with pytest.raises(InternalHitlContinuationCapabilityError):
        require_internal_hitl_continuation(None)


def test_falsey_custom_port_still_used() -> None:
    deps = wire_execution_engine_continuation_dependencies()

    class _ZeroPort:
        def __getattr__(self, name: str) -> object:
            return getattr(deps.continuation, name)

    cap = InternalOrchestrationContinuation(
        port=_ZeroPort(),  # type: ignore[arg-type]
        lifecycle_driver=deps.lifecycle_driver,
    )
    assert cap.port is not None

# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.8-H1-R1-R1 — dispatched execution correlation fail-closed gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from intergrax.autonomous_work.worker_recovery_capability_fulfillment_service import (
    WorkerRecoveryCapabilityFulfillmentService,
)
from intergrax.contracts.autonomous_work.lifecycle import WorkerLifecycleState
from intergrax.contracts.autonomous_work.obstacle_recovery import RecoveryStrategy
from intergrax.contracts.autonomous_work.recovery_orchestration import (
    WorkerRecoveryOrchestrationDisposition,
    WorkerRecoveryOrchestrationResult,
)
from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentDisposition,
    WorkerCapabilityFulfillmentRequest,
    WorkerCapabilityFulfillmentResult,
)
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryProvenance,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityExecutionDisposition,
    WorkerQualifiedCapabilityExecutionResult,
    WorkerQualifiedCapabilityResumeOutcome,
    WorkerQualifiedCapabilityResumeResult,
)
from intergrax.contracts.execution_identity import ExecutionId
from tests.unit.autonomous_work.test_uca6c_r6_r5_8_worker_consumer_e2e import (
    _fulfillment_request,
)
from tests.unit.autonomous_work.test_uca6c_worker_qualified_capability_resume import (
    _EXEC_ID,
)
from tests.unit.autonomous_work.test_worker_recovery_orchestration import (
    _decision,
    _harness,
    _orchestration_request,
)

pytestmark = [pytest.mark.unit]

_NOW = datetime(2026, 3, 20, 12, 0, tzinfo=UTC)
_EXEC_ID_OTHER = ExecutionId("exec_" + "e" * 32)


@dataclass
class _SemanticFulfillmentPort:
    result: WorkerCapabilityFulfillmentResult
    calls: int = 0

    def fulfill(self, request: WorkerCapabilityFulfillmentRequest):
        self.calls += 1
        return self.result


def _provenance() -> WorkerCapabilityRecoveryProvenance:
    return WorkerCapabilityRecoveryProvenance(
        worker_need_id="need",
        canonical_need_id="canonical",
        discovery_correlation_id="corr",
        discovery_completion_outcome="direct_reuse",
        evidence_refs=(),
    )


def _dispatched_execution_result(
    *,
    execution_id: ExecutionId | None,
) -> WorkerQualifiedCapabilityExecutionResult:
    return WorkerQualifiedCapabilityExecutionResult(
        disposition=WorkerQualifiedCapabilityExecutionDisposition.DISPATCHED,
        execution_request_id="worker-qualified-capability-execution:corr",
        execution_id=execution_id,
    )


async def _orchestrate_fulfillment(
    fulfillment_result: WorkerCapabilityFulfillmentResult,
) -> tuple[WorkerRecoveryOrchestrationResult, dict[str, object]]:
    recording = _SemanticFulfillmentPort(result=fulfillment_result)
    service, ctx = _harness()
    service._recovery_capability_fulfillment = (
        WorkerRecoveryCapabilityFulfillmentService(
            fulfillment=recording,
        )
    )
    service._recovery_capability_fulfillment_request_builder = __import__(
        "tests.unit.autonomous_work.test_uca6c_r6_r5_8_worker_consumer_e2e",
        fromlist=["_StaticFulfillmentRequestBuilder"],
    )._StaticFulfillmentRequestBuilder(_fulfillment_request())
    orch = await service.orchestrate(
        _orchestration_request(
            decision=_decision(strategy=RecoveryStrategy.ACQUIRE_CAPABILITY),
        ),
    )
    return orch, ctx


@pytest.mark.asyncio
async def test_execution_dispatched_happy_path_sets_correlation_and_waiting_external() -> (
    None
):
    fulfillment_result = WorkerCapabilityFulfillmentResult(
        disposition=WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED,
        provenance=_provenance(),
        execution_result=_dispatched_execution_result(execution_id=_EXEC_ID),
        decided_at=_NOW,
    )
    orch, ctx = await _orchestrate_fulfillment(fulfillment_result)
    assert orch.disposition is WorkerRecoveryOrchestrationDisposition.ATTEMPT_DISPATCHED
    assert orch.episode.last_execution_id == _EXEC_ID
    assert orch.episode.attempt_count == 1
    assert orch.episode.claimed_attempt_number == 1
    worker = ctx["worker_repo"].get(worker_instance_id=orch.episode.worker_instance_id)
    assert worker is not None
    assert worker.lifecycle_state is WorkerLifecycleState.WAITING_EXTERNAL


@pytest.mark.asyncio
async def test_execution_dispatched_missing_execution_id_fail_closed_direct() -> None:
    fulfillment_result = WorkerCapabilityFulfillmentResult(
        disposition=WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED,
        provenance=_provenance(),
        execution_result=_dispatched_execution_result(execution_id=None),
        decided_at=_NOW,
    )
    orch, ctx = await _orchestrate_fulfillment(fulfillment_result)
    assert orch.disposition is WorkerRecoveryOrchestrationDisposition.ESCALATED
    assert orch.episode.terminal_reason == "capability_fulfillment_execution_id_missing"
    assert orch.episode.last_execution_id is None
    assert orch.episode.attempt_count == 0
    assert orch.episode.claimed_attempt_number is None
    worker = ctx["worker_repo"].get(worker_instance_id=orch.episode.worker_instance_id)
    assert worker is not None
    assert worker.lifecycle_state is WorkerLifecycleState.WORKING


@pytest.mark.asyncio
async def test_execution_dispatched_missing_execution_id_fail_closed_resume() -> None:
    fulfillment_result = WorkerCapabilityFulfillmentResult(
        disposition=WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED,
        provenance=_provenance(),
        resume_result=WorkerQualifiedCapabilityResumeResult(
            outcome=WorkerQualifiedCapabilityResumeOutcome.EXECUTION_DISPATCHED,
            resume_operation_id="resume-op-1",
            provenance=_provenance(),
            execution_result=_dispatched_execution_result(execution_id=None),
        ),
        decided_at=_NOW,
    )
    orch, ctx = await _orchestrate_fulfillment(fulfillment_result)
    assert orch.disposition is WorkerRecoveryOrchestrationDisposition.ESCALATED
    assert orch.episode.terminal_reason == "capability_fulfillment_execution_id_missing"
    assert orch.episode.attempt_count == 0
    worker = ctx["worker_repo"].get(worker_instance_id=orch.episode.worker_instance_id)
    assert worker is not None
    assert worker.lifecycle_state is WorkerLifecycleState.WORKING


@pytest.mark.asyncio
async def test_execution_dispatched_conflicting_execution_id_sources_fail_closed() -> (
    None
):
    fulfillment_result = WorkerCapabilityFulfillmentResult(
        disposition=WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED,
        provenance=_provenance(),
        execution_result=_dispatched_execution_result(execution_id=_EXEC_ID),
        resume_result=WorkerQualifiedCapabilityResumeResult(
            outcome=WorkerQualifiedCapabilityResumeOutcome.EXECUTION_DISPATCHED,
            resume_operation_id="resume-op-2",
            provenance=_provenance(),
            execution_result=_dispatched_execution_result(execution_id=_EXEC_ID_OTHER),
        ),
        decided_at=_NOW,
    )
    orch, ctx = await _orchestrate_fulfillment(fulfillment_result)
    assert orch.disposition is WorkerRecoveryOrchestrationDisposition.ESCALATED
    assert (
        orch.episode.terminal_reason == "capability_fulfillment_execution_id_conflict"
    )
    assert orch.episode.last_execution_id is None
    assert orch.episode.attempt_count == 0
    worker = ctx["worker_repo"].get(worker_instance_id=orch.episode.worker_instance_id)
    assert worker is not None
    assert worker.lifecycle_state is WorkerLifecycleState.WORKING

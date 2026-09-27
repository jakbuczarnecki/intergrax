# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime

import pytest

from intergrax.autonomous_work.worker_capability_fulfillment_coordinator import (
    WorkerCapabilityFulfillmentCoordinator,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityNeedKind,
    WorkerCapabilityAcquisitionRequest,
    WorkerCapabilityNeed,
)
from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentDisposition,
    WorkerCapabilityFulfillmentRequest,
)
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryOutcome,
    WorkerCapabilityRecoveryPhase,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityResumeOutcome,
    WorkerQualifiedCapabilityResumeResult,
    derive_qualified_capability_execution_request_id,
    derive_worker_capability_resume_operation_id,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    derive_qualified_capability_binding_operation_id,
)
from intergrax.contracts.capability_qualification.qualified_subject import (
    qualified_capability_subject_from_result,
)
from intergrax.contracts.execution_identity import TaskId
from intergrax.contracts.tools.qualified_capability_execution_intent_preparation import (
    QualifiedCapabilityExecutionIntentPreparationOutcome,
    QualifiedCapabilityExecutionIntentPreparationRequest,
    QualifiedCapabilityExecutionIntentPreparationResult,
)
from tests.unit.autonomous_work.test_uca6b_worker_capability_recovery import (
    _PROFILE,
    _WORKER_ID,
    _recovery_decision,
)
from tests.unit.autonomous_work.test_uca6c_r_production_resume import (
    _READ,
    _acquisition,
    _provenance,
    _qualification,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 3, 26, 12, 0, tzinfo=UTC)
_TASK_ID = TaskId("task_00000000000000000000000000000001")
_RECOVERY_DECISION = "recovery:intent-hook"


def _need() -> WorkerCapabilityNeed:
    return WorkerCapabilityNeed(
        worker_instance_id=_WORKER_ID,
        obstacle_id=f"{_WORKER_ID}:obstacle:intent",
        need_kind=CapabilityNeedKind.TOOL_OPERATION,
        required_operations=("invoke",),
        capability_profile_ref=_PROFILE,
        requested_at=_NOW,
        recovery_decision_id=_RECOVERY_DECISION,
    )


def _fulfillment_request() -> WorkerCapabilityFulfillmentRequest:
    need = _need()
    decision = replace(_recovery_decision(need), decision_id=_RECOVERY_DECISION)
    return WorkerCapabilityFulfillmentRequest(
        acquisition_request=WorkerCapabilityAcquisitionRequest(
            need=need,
            recovery_decision=decision,
            capability_profile_ref=_PROFILE,
        ),
        worker_instance_id=_WORKER_ID,
        tenant_id="tenant-1",
        task_id=_TASK_ID,
        requested_at=_NOW,
        requested_authority_scopes=(_READ,),
        allow_generic_acquisition=False,
    )


@dataclass
class _RecoveryPort:
    outcome: WorkerCapabilityRecoveryOutcome

    def coordinate_recovery(self, request, *, decided_at, allow_generic_acquisition):
        return self.outcome


@dataclass
class _ResumePort:
    calls: int = 0

    def resume(self, request, *, decided_at=None):
        self.calls += 1
        return WorkerQualifiedCapabilityResumeResult(
            outcome=WorkerQualifiedCapabilityResumeOutcome.EXECUTION_DISPATCHED,
            resume_operation_id=request.resume_operation_id,
            provenance=_provenance(),
            decided_at=_NOW,
        )

    async def resume_async(self, request, *, decided_at=None):
        return self.resume(request, decided_at=decided_at)


@dataclass
class _DirectReusePort:
    def fulfill(self, request):
        raise AssertionError("not used")


@dataclass
class _IntentPreparation:
    outcome: QualifiedCapabilityExecutionIntentPreparationOutcome

    def prepare(
        self,
        request: QualifiedCapabilityExecutionIntentPreparationRequest,
    ) -> QualifiedCapabilityExecutionIntentPreparationResult:
        return QualifiedCapabilityExecutionIntentPreparationResult(outcome=self.outcome)


def _coordinator(
    *,
    preparation: _IntentPreparation | None,
) -> tuple[WorkerCapabilityFulfillmentCoordinator, _ResumePort]:
    recovery = _RecoveryPort(
        outcome=WorkerCapabilityRecoveryOutcome(
            phase=WorkerCapabilityRecoveryPhase.QUALIFICATION_COMPLETE,
            provenance=_provenance(),
            acquisition_result=_acquisition(),
            qualification_result=_qualification(),
            decided_at=_NOW,
        ),
    )
    resume = _ResumePort()
    coordinator = WorkerCapabilityFulfillmentCoordinator(
        recovery=recovery,
        resume=resume,
        direct_reuse=_DirectReusePort(),
        intent_preparation=preparation,
    )
    return coordinator, resume


def test_no_preparation_provider_resume_called() -> None:
    coordinator, resume = _coordinator(preparation=None)
    result = coordinator.fulfill(_fulfillment_request())
    assert resume.calls == 1
    assert result.disposition is not WorkerCapabilityFulfillmentDisposition.FAIL_CLOSED


@pytest.mark.parametrize(
    "outcome",
    [
        QualifiedCapabilityExecutionIntentPreparationOutcome.NOT_APPLICABLE,
        QualifiedCapabilityExecutionIntentPreparationOutcome.CREATED,
        QualifiedCapabilityExecutionIntentPreparationOutcome.ALREADY_RECORDED_IDENTICAL,
    ],
)
def test_preparation_success_resume_called(outcome) -> None:
    coordinator, resume = _coordinator(preparation=_IntentPreparation(outcome))
    coordinator.fulfill(_fulfillment_request())
    assert resume.calls == 1


@pytest.mark.parametrize(
    "outcome",
    [
        QualifiedCapabilityExecutionIntentPreparationOutcome.UNAVAILABLE,
        QualifiedCapabilityExecutionIntentPreparationOutcome.CONFLICT,
        QualifiedCapabilityExecutionIntentPreparationOutcome.INTEGRITY_FAILURE,
        QualifiedCapabilityExecutionIntentPreparationOutcome.INVALID_OPERATION,
    ],
)
def test_preparation_failure_fail_closed_no_resume(outcome) -> None:
    coordinator, resume = _coordinator(preparation=_IntentPreparation(outcome))
    result = coordinator.fulfill(_fulfillment_request())
    assert result.disposition is WorkerCapabilityFulfillmentDisposition.FAIL_CLOSED
    assert resume.calls == 0


def test_predicted_execution_request_id_matches_resume_coordinator_derivation() -> None:
    qualification = _qualification()
    subject = qualified_capability_subject_from_result(qualification)
    assert subject is not None
    resume_id = derive_worker_capability_resume_operation_id(
        recovery_decision_id=_RECOVERY_DECISION,
        qualification_request_id=qualification.qualification_request_id,
    )
    binding_id = derive_qualified_capability_binding_operation_id(
        resume_operation_id=resume_id,
        qualified_subject_reference=subject.qualified_subject_reference,
    )
    expected = derive_qualified_capability_execution_request_id(
        resume_operation_id=resume_id,
        binding_operation_id=binding_id,
    )
    captured: list[str] = []

    @dataclass
    class _CapturingPreparation:
        def prepare(self, request):
            captured.append(request.execution_request_id)
            return QualifiedCapabilityExecutionIntentPreparationResult(
                outcome=QualifiedCapabilityExecutionIntentPreparationOutcome.CREATED,
            )

    coordinator, _ = _coordinator(preparation=_CapturingPreparation())
    coordinator.fulfill(_fulfillment_request())
    assert captured == [expected]


@pytest.mark.asyncio
async def test_async_same_semantics_as_sync() -> None:
    coordinator, resume = _coordinator(
        preparation=_IntentPreparation(
            QualifiedCapabilityExecutionIntentPreparationOutcome.UNAVAILABLE,
        ),
    )
    result = await coordinator.fulfill_async(_fulfillment_request())
    assert result.disposition is WorkerCapabilityFulfillmentDisposition.FAIL_CLOSED
    assert resume.calls == 0

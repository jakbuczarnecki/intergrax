# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from intergrax.autonomous_work.worker_capability_fulfillment_coordinator import (
    WorkerCapabilityFulfillmentCoordinator,
)
from intergrax.autonomous_work.worker_capability_fulfillment_ports import (
    WorkerQualifiedCapabilityResumePort,
)
from tests.unit.autonomous_work.test_uca6c_worker_qualified_capability_resume import (
    _RecordingBindingProvider,
    _coordinator as _resume_coordinator,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityNeedKind,
    WorkerCapabilityAcquisitionRequest,
    WorkerCapabilityNeed,
)
from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentRequest,
)
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryOutcome,
    WorkerCapabilityRecoveryPhase,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityResumeOutcome,
    WorkerQualifiedCapabilityResumeRequest,
    WorkerQualifiedCapabilityResumeResult,
)
from intergrax.contracts.execution_identity import TaskId
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

_NOW = datetime(2026, 3, 27, 10, 0, tzinfo=UTC)
_TASK_ID = TaskId("task_00000000000000000000000000000001")
_RECOVERY_DECISION = "recovery:r1-contract"


def _need() -> WorkerCapabilityNeed:
    return WorkerCapabilityNeed(
        worker_instance_id=_WORKER_ID,
        obstacle_id=f"{_WORKER_ID}:obstacle:r1",
        need_kind=CapabilityNeedKind.TOOL_OPERATION,
        required_operations=("invoke",),
        capability_profile_ref=_PROFILE,
        requested_at=_NOW,
        recovery_decision_id=_RECOVERY_DECISION,
    )


def _fulfillment_request() -> WorkerCapabilityFulfillmentRequest:
    need = _need()
    decision = _recovery_decision(need)
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


def _qualified_recovery() -> WorkerCapabilityRecoveryOutcome:
    return WorkerCapabilityRecoveryOutcome(
        phase=WorkerCapabilityRecoveryPhase.QUALIFICATION_COMPLETE,
        provenance=_provenance(),
        acquisition_result=_acquisition(),
        qualification_result=_qualification(),
        decided_at=_NOW,
    )


def _resume_result(
    request: WorkerQualifiedCapabilityResumeRequest,
) -> WorkerQualifiedCapabilityResumeResult:
    return WorkerQualifiedCapabilityResumeResult(
        outcome=WorkerQualifiedCapabilityResumeOutcome.EXECUTION_DISPATCHED,
        resume_operation_id=request.resume_operation_id,
        provenance=request.provenance,
        decided_at=_NOW,
    )


@dataclass
class _SyncOnlyResumeProvider:
    def resume(
        self,
        request: WorkerQualifiedCapabilityResumeRequest,
        *,
        decided_at: datetime | None = None,
    ) -> WorkerQualifiedCapabilityResumeResult:
        return _resume_result(request)


@dataclass
class _CustomResumeProvider:
    resume_calls: int = 0
    resume_async_calls: int = 0
    async_resume_requests: list[WorkerQualifiedCapabilityResumeRequest] = field(
        default_factory=list
    )

    def resume(
        self,
        request: WorkerQualifiedCapabilityResumeRequest,
        *,
        decided_at: datetime | None = None,
    ) -> WorkerQualifiedCapabilityResumeResult:
        self.resume_calls += 1
        return _resume_result(request)

    async def resume_async(
        self,
        request: WorkerQualifiedCapabilityResumeRequest,
        *,
        decided_at: datetime | None = None,
    ) -> WorkerQualifiedCapabilityResumeResult:
        self.resume_async_calls += 1
        self.async_resume_requests.append(request)
        return _resume_result(request)


@dataclass
class _RecoveryPort:
    outcome: WorkerCapabilityRecoveryOutcome

    def coordinate_recovery(
        self,
        request: WorkerCapabilityAcquisitionRequest,
        *,
        decided_at: datetime | None = None,
        allow_generic_acquisition: bool = True,
    ) -> WorkerCapabilityRecoveryOutcome:
        return self.outcome


@dataclass
class _DirectReusePort:
    def fulfill_direct_reuse(self, request, recovery):
        raise AssertionError("not used")


def _fulfillment_coordinator(
    resume: _CustomResumeProvider,
) -> WorkerCapabilityFulfillmentCoordinator:
    return WorkerCapabilityFulfillmentCoordinator(
        recovery=_RecoveryPort(outcome=_qualified_recovery()),
        resume=resume,
        direct_reuse=_DirectReusePort(),
        intent_preparation=None,
    )


def test_protocol_exposes_resume_and_resume_async() -> None:
    assert hasattr(WorkerQualifiedCapabilityResumePort, "resume")
    assert hasattr(WorkerQualifiedCapabilityResumePort, "resume_async")


def test_custom_resume_provider_satisfies_protocol() -> None:
    assert isinstance(_CustomResumeProvider(), WorkerQualifiedCapabilityResumePort)


def test_sync_only_resume_provider_does_not_satisfy_protocol() -> None:
    assert not isinstance(_SyncOnlyResumeProvider(), WorkerQualifiedCapabilityResumePort)


def test_fulfill_qualified_path_calls_sync_resume_only() -> None:
    resume = _CustomResumeProvider()
    coordinator = _fulfillment_coordinator(resume)
    coordinator.fulfill(_fulfillment_request())
    assert resume.resume_calls == 1
    assert resume.resume_async_calls == 0


@pytest.mark.asyncio
async def test_fulfill_async_qualified_path_calls_async_resume_only() -> None:
    resume = _CustomResumeProvider()
    coordinator = _fulfillment_coordinator(resume)
    await coordinator.fulfill_async(_fulfillment_request())
    assert resume.resume_calls == 0
    assert resume.resume_async_calls == 1


def test_resume_coordinator_satisfies_protocol() -> None:
    coordinator = _resume_coordinator(_RecordingBindingProvider())
    assert isinstance(coordinator, WorkerQualifiedCapabilityResumePort)

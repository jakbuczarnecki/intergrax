# © Artur Czarnecki. All rights reserved.

"""AW-7B — ephemeral capability execution service tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from intergrax.autonomous_work.ephemeral_capability_execution import (
    WorkerEphemeralCapabilityExecutionService,
)
from intergrax.autonomous_work.ephemeral_capability_execution_ports import (
    WorkerEphemeralCapabilityExecutionPort,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    ACQUISITION_DECISION_POLICY_VERSION,
    CapabilityAcquisitionDisposition,
    CapabilityAcquisitionReasonCode,
    CapabilityNeedKind,
    WorkerAutonomyLevel,
    WorkerCapabilityAcquisitionDecision,
    WorkerCapabilityCandidate,
    WorkerCapabilityCandidateKind,
    WorkerCapabilityNeed,
    derive_worker_capability_acquisition_decision_id,
    derive_worker_capability_need_id,
)
from intergrax.contracts.autonomous_work.ephemeral_capability_execution import (
    WorkerEphemeralCapabilityExecutionCorrelation,
    WorkerEphemeralCapabilityExecutionReasonCode,
    WorkerEphemeralCapabilityExecutionRequest,
    WorkerEphemeralCapabilityExecutionResult,
    WorkerEphemeralCapabilityExecutionStatus,
    WorkerEphemeralCapabilityReference,
)
from intergrax.contracts.autonomous_work.profile_reference import (
    CapabilityProfileRef,
    CodecraftProfileRef,
    initial_profile_version,
)
from intergrax.contracts.autonomous_work.references import ProblemReference
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = pytest.mark.unit

_UTC = UTC
_NOW = datetime(2026, 9, 7, 10, 0, tzinfo=_UTC)
_WORKER_ID = contract_suite.mint_worker_instance_id()
_EVIDENCE = ProblemReference("problem/evidence/a1-service-1")
_CAPABILITY_PROFILE = CapabilityProfileRef(
    profile_id="cap/default",
    version=initial_profile_version(),
)
_CODECRAFT_PROFILE = CodecraftProfileRef(
    profile_id="codecraft/default",
    version=initial_profile_version(),
)
_OPERATION = "document.parse_csv"


@dataclass
class RecordingEphemeralExecutionPort:
    calls: list[WorkerEphemeralCapabilityExecutionRequest] = field(default_factory=list)
    result: WorkerEphemeralCapabilityExecutionResult | None = None

    def execute(
        self,
        request: WorkerEphemeralCapabilityExecutionRequest,
    ) -> WorkerEphemeralCapabilityExecutionResult:
        self.calls.append(request)
        assert self.result is not None
        return self.result


def _need() -> WorkerCapabilityNeed:
    return WorkerCapabilityNeed(
        worker_instance_id=_WORKER_ID,
        obstacle_id=f"{_WORKER_ID}:capability/missing:1:occurrence-1",
        need_kind=CapabilityNeedKind.TOOL_OPERATION,
        required_operations=(_OPERATION,),
        capability_profile_ref=_CAPABILITY_PROFILE,
        requested_at=_NOW,
        recovery_decision_id="recovery-a1-1",
        evidence_refs=(_EVIDENCE,),
        codecraft_profile_ref=_CODECRAFT_PROFILE,
    )


def _a1_candidate() -> WorkerCapabilityCandidate:
    return WorkerCapabilityCandidate(
        candidate_id="CODECRAFT_EPHEMERAL:ephemeral:codecraft",
        candidate_kind=WorkerCapabilityCandidateKind.CODECRAFT_EPHEMERAL,
        capability_ref="ephemeral:codecraft",
        source_domain="autonomous_work",
        operations=(_OPERATION,),
        risk_class=WorkerAutonomyLevel.A1_EPHEMERAL_SAFE,
        evidence_refs=(_EVIDENCE,),
        discovered_at=_NOW,
    )


def _a1_decision() -> WorkerCapabilityAcquisitionDecision:
    need = _need()
    selected = _a1_candidate()
    need_id = derive_worker_capability_need_id(need)
    return WorkerCapabilityAcquisitionDecision(
        decision_id=derive_worker_capability_acquisition_decision_id(
            worker_instance_id=_WORKER_ID,
            obstacle_id=need.obstacle_id,
            recovery_decision_id=need.recovery_decision_id,
            need_id=need_id,
            capability_profile_version=need.capability_profile_ref.version.value,
            selected_candidate_id=selected.candidate_id,
            decision_policy_version=ACQUISITION_DECISION_POLICY_VERSION,
        ),
        worker_instance_id=_WORKER_ID,
        obstacle_id=need.obstacle_id,
        recovery_decision_id=need.recovery_decision_id,
        need_id=need_id,
        disposition=CapabilityAcquisitionDisposition.EPHEMERAL_GENERATION_CANDIDATE,
        selected_candidate=selected,
        autonomy_level=WorkerAutonomyLevel.A1_EPHEMERAL_SAFE,
        capability_profile_ref=_CAPABILITY_PROFILE,
        codecraft_profile_ref=_CODECRAFT_PROFILE,
        reason_code=CapabilityAcquisitionReasonCode.A1_CANDIDATE_ALLOWED,
        evidence_refs=(_EVIDENCE,),
        decided_at=_NOW,
    )


def _execution_request(
    decision: WorkerCapabilityAcquisitionDecision | None = None,
) -> WorkerEphemeralCapabilityExecutionRequest:
    resolved = decision or _a1_decision()
    assert resolved.selected_candidate is not None
    return WorkerEphemeralCapabilityExecutionRequest(
        worker_instance_id=_WORKER_ID,
        acquisition_decision=resolved,
        recovery_decision_id=resolved.recovery_decision_id,
        obstacle_id=resolved.obstacle_id,
        need_id=resolved.need_id,
        selected_candidate=resolved.selected_candidate,
        codecraft_profile_ref=_CODECRAFT_PROFILE,
        generation_goal="parse csv helper",
        required_operations=(_OPERATION,),
        correlation=WorkerEphemeralCapabilityExecutionCorrelation(
            tenant_id="tenant-a",
            task_id="task-a",
        ),
        requested_at=_NOW,
        evidence_refs=(_EVIDENCE,),
    )


def _port_result(
    *,
    status: WorkerEphemeralCapabilityExecutionStatus,
    reason_code: WorkerEphemeralCapabilityExecutionReasonCode,
) -> WorkerEphemeralCapabilityExecutionResult:
    base = {
        "status": status,
        "reason_code": reason_code,
        "worker_instance_id": _WORKER_ID,
        "acquisition_decision_id": _a1_decision().decision_id,
        "need_id": derive_worker_capability_need_id(_need()),
        "evidence_refs": (_EVIDENCE,),
        "executed_at": _NOW,
    }
    if status is WorkerEphemeralCapabilityExecutionStatus.SUCCEEDED:
        return WorkerEphemeralCapabilityExecutionResult(
            **base,
            ephemeral_capability=WorkerEphemeralCapabilityReference(
                craft_id="craft_test",
                ephemeral_tool_id="ephemeral.craft_test.helper",
            ),
            craft_correlation="craft_test",
        )
    return WorkerEphemeralCapabilityExecutionResult(**base, error_code=reason_code.value)


def test_valid_a1_calls_port_once() -> None:
    port = RecordingEphemeralExecutionPort(
        result=_port_result(
            status=WorkerEphemeralCapabilityExecutionStatus.SUCCEEDED,
            reason_code=WorkerEphemeralCapabilityExecutionReasonCode.A1_EXECUTION_SUCCEEDED,
        ),
    )
    service = WorkerEphemeralCapabilityExecutionService(execution_port=port)
    request = _execution_request()
    result = service.execute(request)

    assert len(port.calls) == 1
    assert port.calls[0] is request
    assert result.status is WorkerEphemeralCapabilityExecutionStatus.SUCCEEDED


def test_invalid_disposition_does_not_call_port() -> None:
    need = _need()
    need_id = derive_worker_capability_need_id(need)
    a2_candidate = WorkerCapabilityCandidate(
        candidate_id="ADAPTIVE_INTEGRATION:adaptive:integration",
        candidate_kind=WorkerCapabilityCandidateKind.ADAPTIVE_INTEGRATION,
        capability_ref="adaptive:integration",
        source_domain="autonomous_work",
        operations=(_OPERATION,),
        risk_class=WorkerAutonomyLevel.A2_SCOPED_ADAPTIVE,
        evidence_refs=(_EVIDENCE,),
        discovered_at=_NOW,
    )
    bad = WorkerCapabilityAcquisitionDecision(
        decision_id=derive_worker_capability_acquisition_decision_id(
            worker_instance_id=_WORKER_ID,
            obstacle_id=need.obstacle_id,
            recovery_decision_id=need.recovery_decision_id,
            need_id=need_id,
            capability_profile_version=need.capability_profile_ref.version.value,
            selected_candidate_id=a2_candidate.candidate_id,
            decision_policy_version=ACQUISITION_DECISION_POLICY_VERSION,
        ),
        worker_instance_id=_WORKER_ID,
        obstacle_id=need.obstacle_id,
        recovery_decision_id=need.recovery_decision_id,
        need_id=need_id,
        disposition=CapabilityAcquisitionDisposition.SCOPED_ADAPTATION_CANDIDATE,
        selected_candidate=a2_candidate,
        autonomy_level=WorkerAutonomyLevel.A2_SCOPED_ADAPTIVE,
        capability_profile_ref=_CAPABILITY_PROFILE,
        reason_code=CapabilityAcquisitionReasonCode.A2_ADAPTATION_REQUIRED,
        evidence_refs=(_EVIDENCE,),
        decided_at=_NOW,
    )
    port = RecordingEphemeralExecutionPort(
        result=_port_result(
            status=WorkerEphemeralCapabilityExecutionStatus.SUCCEEDED,
            reason_code=WorkerEphemeralCapabilityExecutionReasonCode.A1_EXECUTION_SUCCEEDED,
        ),
    )
    service = WorkerEphemeralCapabilityExecutionService(execution_port=port)
    result = service.execute(_execution_request(bad))

    assert port.calls == []
    assert result.status is WorkerEphemeralCapabilityExecutionStatus.DENIED


def test_worker_mismatch_does_not_call_port() -> None:
    port = RecordingEphemeralExecutionPort(
        result=_port_result(
            status=WorkerEphemeralCapabilityExecutionStatus.SUCCEEDED,
            reason_code=WorkerEphemeralCapabilityExecutionReasonCode.A1_EXECUTION_SUCCEEDED,
        ),
    )
    service = WorkerEphemeralCapabilityExecutionService(execution_port=port)
    request = _execution_request()
    other_worker = contract_suite.mint_worker_instance_id()
    mismatched = WorkerEphemeralCapabilityExecutionRequest(
        worker_instance_id=other_worker,
        acquisition_decision=request.acquisition_decision,
        recovery_decision_id=request.recovery_decision_id,
        obstacle_id=request.obstacle_id,
        need_id=request.need_id,
        selected_candidate=request.selected_candidate,
        codecraft_profile_ref=request.codecraft_profile_ref,
        generation_goal=request.generation_goal,
        required_operations=request.required_operations,
        correlation=request.correlation,
        requested_at=request.requested_at,
        evidence_refs=request.evidence_refs,
    )
    result = service.execute(mismatched)

    assert port.calls == []
    assert result.status is WorkerEphemeralCapabilityExecutionStatus.CONFLICT


@pytest.mark.parametrize(
    ("port_status", "port_reason"),
    [
        (WorkerEphemeralCapabilityExecutionStatus.DENIED, WorkerEphemeralCapabilityExecutionReasonCode.PROVIDER_DENIED),
        (
            WorkerEphemeralCapabilityExecutionStatus.PENDING_HITL,
            WorkerEphemeralCapabilityExecutionReasonCode.PROVIDER_PENDING_HITL,
        ),
        (WorkerEphemeralCapabilityExecutionStatus.FAILED, WorkerEphemeralCapabilityExecutionReasonCode.PROVIDER_FAILED),
        (
            WorkerEphemeralCapabilityExecutionStatus.UNAVAILABLE,
            WorkerEphemeralCapabilityExecutionReasonCode.PROVIDER_UNAVAILABLE,
        ),
        (WorkerEphemeralCapabilityExecutionStatus.SUCCEEDED, WorkerEphemeralCapabilityExecutionReasonCode.A1_EXECUTION_SUCCEEDED),
    ],
)
def test_port_outcomes_preserved(
    port_status: WorkerEphemeralCapabilityExecutionStatus,
    port_reason: WorkerEphemeralCapabilityExecutionReasonCode,
) -> None:
    port = RecordingEphemeralExecutionPort(result=_port_result(status=port_status, reason_code=port_reason))
    service = WorkerEphemeralCapabilityExecutionService(execution_port=port)
    result = service.execute(_execution_request())
    assert result.status is port_status
    assert result.reason_code is port_reason


def test_fake_port_does_not_import_codecraft() -> None:
    assert isinstance(RecordingEphemeralExecutionPort(), WorkerEphemeralCapabilityExecutionPort)

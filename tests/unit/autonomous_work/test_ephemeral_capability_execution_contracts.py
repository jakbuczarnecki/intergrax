# © Artur Czarnecki. All rights reserved.

"""AW-7B — ephemeral capability execution contract tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

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
    validate_a1_ephemeral_execution_eligibility,
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
_EVIDENCE = ProblemReference("problem/evidence/a1-1")
_CAPABILITY_PROFILE = CapabilityProfileRef(
    profile_id="cap/default",
    version=initial_profile_version(),
)
_CODECRAFT_PROFILE = CodecraftProfileRef(
    profile_id="codecraft/default",
    version=initial_profile_version(),
)
_OPERATION = "document.parse_csv"


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


def _a1_decision(candidate: WorkerCapabilityCandidate | None = None) -> WorkerCapabilityAcquisitionDecision:
    need = _need()
    selected = candidate or _a1_candidate()
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


def test_a1_request_immutability_fields() -> None:
    request = _execution_request()
    assert request.worker_instance_id == _WORKER_ID
    assert request.codecraft_profile_ref == _CODECRAFT_PROFILE


def test_validate_a1_eligibility_accepts_valid_request() -> None:
    assert validate_a1_ephemeral_execution_eligibility(_execution_request()) is None


def test_validate_a1_rejects_wrong_disposition() -> None:
    need = _need()
    need_id = derive_worker_capability_need_id(need)
    a3_candidate = WorkerCapabilityCandidate(
        candidate_id="DURABLE_PRODUCTION_CHANGE:durable:production-change",
        candidate_kind=WorkerCapabilityCandidateKind.DURABLE_PRODUCTION_CHANGE,
        capability_ref="durable:production-change",
        source_domain="autonomous_work",
        operations=(_OPERATION,),
        risk_class=WorkerAutonomyLevel.A3_PRODUCTION_CHANGE,
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
            selected_candidate_id=a3_candidate.candidate_id,
            decision_policy_version=ACQUISITION_DECISION_POLICY_VERSION,
        ),
        worker_instance_id=_WORKER_ID,
        obstacle_id=need.obstacle_id,
        recovery_decision_id=need.recovery_decision_id,
        need_id=need_id,
        disposition=CapabilityAcquisitionDisposition.PRODUCTION_CHANGE_REQUIRED,
        selected_candidate=a3_candidate,
        autonomy_level=WorkerAutonomyLevel.A3_PRODUCTION_CHANGE,
        capability_profile_ref=_CAPABILITY_PROFILE,
        reason_code=CapabilityAcquisitionReasonCode.A3_PRODUCTION_CHANGE_REQUIRED,
        evidence_refs=(_EVIDENCE,),
        decided_at=_NOW,
    )
    request = _execution_request(bad)
    assert (
        validate_a1_ephemeral_execution_eligibility(request)
        is WorkerEphemeralCapabilityExecutionReasonCode.DISPOSITION_MISMATCH
    )


def test_validate_a1_rejects_a2_candidate_kind() -> None:
    candidate = WorkerCapabilityCandidate(
        candidate_id="ADAPTIVE_INTEGRATION:adaptive:integration",
        candidate_kind=WorkerCapabilityCandidateKind.ADAPTIVE_INTEGRATION,
        capability_ref="adaptive:integration",
        source_domain="autonomous_work",
        operations=(_OPERATION,),
        risk_class=WorkerAutonomyLevel.A2_SCOPED_ADAPTIVE,
        evidence_refs=(_EVIDENCE,),
        discovered_at=_NOW,
    )
    decision = _a1_decision(candidate)
    request = _execution_request(decision)
    assert (
        validate_a1_ephemeral_execution_eligibility(request)
        is WorkerEphemeralCapabilityExecutionReasonCode.CANDIDATE_KIND_MISMATCH
    )


def test_validate_a1_rejects_worker_mismatch() -> None:
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
    assert (
        validate_a1_ephemeral_execution_eligibility(mismatched)
        is WorkerEphemeralCapabilityExecutionReasonCode.CORRELATION_CONFLICT
    )


def test_succeeded_result_requires_ephemeral_capability() -> None:
    with pytest.raises(ValueError, match="SUCCEEDED requires ephemeral_capability"):
        WorkerEphemeralCapabilityExecutionResult(
            status=WorkerEphemeralCapabilityExecutionStatus.SUCCEEDED,
            reason_code=WorkerEphemeralCapabilityExecutionReasonCode.A1_EXECUTION_SUCCEEDED,
            worker_instance_id=_WORKER_ID,
            acquisition_decision_id="decision-a1",
            need_id="need-a1",
            evidence_refs=(_EVIDENCE,),
            executed_at=_NOW,
            craft_correlation="craft_123",
        )


def test_succeeded_result_accepts_ephemeral_reference() -> None:
    result = WorkerEphemeralCapabilityExecutionResult(
        status=WorkerEphemeralCapabilityExecutionStatus.SUCCEEDED,
        reason_code=WorkerEphemeralCapabilityExecutionReasonCode.A1_EXECUTION_SUCCEEDED,
        worker_instance_id=_WORKER_ID,
        acquisition_decision_id="decision-a1",
        need_id="need-a1",
        evidence_refs=(_EVIDENCE,),
        executed_at=_NOW,
        ephemeral_capability=WorkerEphemeralCapabilityReference(
            craft_id="craft_123",
            ephemeral_tool_id="ephemeral.craft_123.helper",
        ),
        craft_correlation="craft_123",
    )
    assert result.ephemeral_capability is not None

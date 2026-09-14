# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.decision_authoritative_exposure import (
    DecisionEvaluationScope,
    ExposureAccepted,
    ExposureResolution,
)
from intergrax.contracts.decision_human_review import (
    decision_human_review_request,
    verification_challenged_human_review_reason,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    DecisionVersionLineage,
    candidate_decision,
    candidate_decision_ref,
    decision_lineage_ref,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_resolution import (
    AuthoritativeResolutionRecord,
    DecisionResolution,
)
from intergrax.contracts.decision_verification import (
    VerificationDisposition,
    VerificationStageOutcome,
    verification_result,
    verification_stage_record,
    validate_verification_stage_kind,
)
from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
from intergrax.runtime.decision_exposure_mapping import (
    DecisionFlowExposureMappingError,
    decision_flow_result_to_authoritative_exposure,
    decision_flow_scope_to_evaluation_scope,
)
from intergrax.runtime.decision_flow import (
    DecisionFlowHostAction,
    DecisionFlowResult,
    DecisionFlowScope,
)
from intergrax.runtime.decision_human_review import request_decision_human_review
from intergrax.runtime.execution.decision_lifecycle_host import CanonicalDecisionLifecycleHost

pytestmark = pytest.mark.unit


@dataclass
class _Payload:
    text: str


def _candidate() -> object:
    artifact_kind = validate_decision_artifact_kind("agent.execution.result")
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="map", subject="subject-1"),
        tenant_id="tenant-1",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
        ),
    )
    return candidate_decision(
        identity=identity,
        artifact_kind=artifact_kind,
        payload=_Payload(text="ok"),
    )


def _flow_result(
    *,
    host_action: DecisionFlowHostAction,
    accepted: AuthoritativeAcceptedDecision[_Payload] | None = None,
    resolution: AuthoritativeResolutionRecord | None = None,
    flow_scope: DecisionFlowScope = DecisionFlowScope.GRAPH_FINAL,
    human_review_pending: object | None = None,
) -> DecisionFlowResult[_Payload]:
    candidate = _candidate()
    proposal_ref = candidate_decision_ref(candidate)
    verification = verification_result(
        proposal_ref=proposal_ref,
        disposition=VerificationDisposition.PASSED,
        stage_records=(
            verification_stage_record(
                proposal_ref=proposal_ref,
                stage=validate_verification_stage_kind("structural"),
                outcome=VerificationStageOutcome.PASSED,
            ),
        ),
    )
    lifecycle_host = CanonicalDecisionLifecycleHost()
    lifecycle_state = lifecycle_host.start(candidate.identity)
    return DecisionFlowResult(
        host_action=host_action,
        flow_scope=flow_scope,
        candidate=candidate,
        verification_result=verification,
        lifecycle_state=lifecycle_state,
        accepted_decision=accepted,
        resolution_record=resolution,
        human_review_pending=human_review_pending,
    )


def _accepted_for_candidate(candidate) -> AuthoritativeAcceptedDecision[_Payload]:
    return AuthoritativeAcceptedDecision(
        identity=candidate.identity,
        artifact=candidate.artifact,
        lineage=DecisionVersionLineage(
            current=decision_lineage_ref(candidate.identity.version),
        ),
    )


def test_m1_accepted_maps_to_exposure_accepted() -> None:
    candidate = _candidate()
    accepted = _accepted_for_candidate(candidate)
    result = _flow_result(
        host_action=DecisionFlowHostAction.CONTINUE,
        accepted=accepted,
    )
    exposure = decision_flow_result_to_authoritative_exposure(result)
    assert type(exposure) is ExposureAccepted
    assert exposure.scope is DecisionEvaluationScope.GRAPH_FINAL
    assert exposure.accepted is accepted


def test_m2_resolution_maps_to_exposure_resolution() -> None:
    candidate = _candidate()
    resolution = AuthoritativeResolutionRecord(
        identity=candidate.identity,
        resolution=DecisionResolution.REJECTED,
    )
    result = _flow_result(
        host_action=DecisionFlowHostAction.BLOCK,
        resolution=resolution,
    )
    exposure = decision_flow_result_to_authoritative_exposure(result)
    assert type(exposure) is ExposureResolution
    assert exposure.resolution is resolution


def test_m3_accepted_and_resolution_fail_closed() -> None:
    candidate = _candidate()
    accepted = _accepted_for_candidate(candidate)
    resolution = AuthoritativeResolutionRecord(
        identity=candidate.identity,
        resolution=DecisionResolution.UNRESOLVED,
    )
    result = _flow_result(
        host_action=DecisionFlowHostAction.CONTINUE,
        accepted=accepted,
        resolution=resolution,
    )
    with pytest.raises(DecisionFlowExposureMappingError):
        decision_flow_result_to_authoritative_exposure(result)


def test_m4_pending_human_returns_no_terminal_exposure() -> None:
    candidate = _candidate()
    request = decision_human_review_request(
        proposal_ref=candidate_decision_ref(candidate),
        reason_code=verification_challenged_human_review_reason(),
    )
    pending = request_decision_human_review(request)
    result = _flow_result(
        host_action=DecisionFlowHostAction.PENDING_HUMAN,
        human_review_pending=pending,
    )
    assert decision_flow_result_to_authoritative_exposure(result) is None


def test_m5_scope_mapping() -> None:
    assert (
        decision_flow_scope_to_evaluation_scope(DecisionFlowScope.GRAPH_FINAL)
        is DecisionEvaluationScope.GRAPH_FINAL
    )
    assert (
        decision_flow_scope_to_evaluation_scope(DecisionFlowScope.UAEP_STEP)
        is DecisionEvaluationScope.UAEP_STEP
    )


def test_m6_accepted_object_identity_preserved() -> None:
    candidate = _candidate()
    accepted = _accepted_for_candidate(candidate)
    result = _flow_result(
        host_action=DecisionFlowHostAction.CONTINUE,
        accepted=accepted,
        flow_scope=DecisionFlowScope.UAEP_STEP,
    )
    exposure = decision_flow_result_to_authoritative_exposure(result)
    assert type(exposure) is ExposureAccepted
    assert exposure.accepted is result.accepted_decision

# © Artur Czarnecki. All rights reserved.

"""DS-E2E-04 — real HITL pause/resume."""

from __future__ import annotations

from dataclasses import replace

import pytest

from intergrax.contracts.decision_human_review import DecisionHumanReviewOutcome
from intergrax.contracts.decision_revision import decision_revision_policy
from intergrax.contracts.decision_identity import next_decision_version
from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage
from intergrax.contracts.decision_record import (
    candidate_decision,
    candidate_decision_ref,
)
from intergrax.contracts.decision_verification import (
    VerificationStageOutcome,
    validate_verification_finding_code,
    validate_verification_requirement_code,
    validate_verification_stage_kind,
    verification_challenge,
    verification_finding,
    verification_stage_record,
)
from intergrax.contracts.decision_verification_stage import (
    VerificationStageExecutionClass,
    VerificationStageRegistration,
    verification_stage_registry,
)
from intergrax.runtime.decision_flow import DecisionFlowHostAction
from intergrax.runtime.decision_verification import VerificationPipeline
from intergrax.runtime.execution.active_decision_lifecycle_host import (
    bind_active_decision_lifecycle_host,
    reset_active_decision_lifecycle_host,
)

from testing_support.decision_e2e.composition import (
    evaluate_decision_flow,
    mint_qualification_identity,
    run_single_model_producer,
)
from testing_support.decision_e2e.contracts import (
    DecisionE2EProofId,
    DecisionE2EQualificationResult,
    QualificationDisposition,
)
from testing_support.decision_e2e.evidence import decision_identity_evidence
from testing_support.decision_e2e.hitl import DurableDecisionHumanReviewPort

pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.qualification,
    pytest.mark.external_provider,
    pytest.mark.network,
    pytest.mark.no_ci,
    pytest.mark.slow,
]


class ChallengedThenPassStage:
    kind = "semantic"

    @property
    def execution_class(self) -> VerificationStageExecutionClass:
        return VerificationStageExecutionClass.PROBABILISTIC

    async def verify(self, candidate):
        proposal_ref = candidate_decision_ref(candidate)
        finding = verification_finding(
            code=validate_verification_finding_code("verification.test.challenged"),
            message="requires human review",
        )
        return verification_stage_record(
            proposal_ref=proposal_ref,
            stage=validate_verification_stage_kind(self.kind),
            outcome=VerificationStageOutcome.CHALLENGED,
            challenge=verification_challenge(
                proposal_ref=proposal_ref,
                stage=validate_verification_stage_kind(self.kind),
                requirement_code=validate_verification_requirement_code(
                    "verification.test.requirement",
                ),
                finding=finding,
            ),
        )


def _hitl_pipeline() -> VerificationPipeline:
    stage = ChallengedThenPassStage()
    return VerificationPipeline(
        registry=verification_stage_registry(
            (
                VerificationStageRegistration(
                    kind=validate_verification_stage_kind(stage.kind),
                    stage=stage,
                    required=True,
                ),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_ds_e2e_04_hitl_pause_resume(
    decision_e2e_composition,
    decision_e2e_report_collector,
) -> None:
    composition = decision_e2e_composition
    identity = mint_qualification_identity(subject="hitl-qualification")
    hitl_port = DurableDecisionHumanReviewPort()
    gate = composition.build_flow_gate(
        pipeline=_hitl_pipeline(),
        revision_policy=decision_revision_policy(max_revisions=0),
        human_review_port=hitl_port,
    )
    token = bind_active_decision_lifecycle_host(composition.lifecycle_host)
    try:
        payload, _ = await run_single_model_producer(
            composition,
            identity=identity,
            task_message="Return recommendation=review with confidence=medium.",
        )
        pending_result = await evaluate_decision_flow(
            composition,
            gate,
            identity=identity,
            payload=payload,
        )
    finally:
        reset_active_decision_lifecycle_host(token)

    assert pending_result.host_action is DecisionFlowHostAction.PENDING_HUMAN
    assert pending_result.human_review_pending is not None
    assert len(hitl_port.pending_requests) == 1
    request = hitl_port.pending_requests[0]

    approve_decision = hitl_port.submit_decision(
        request=request,
        outcome=DecisionHumanReviewOutcome.APPROVED,
    )
    assert approve_decision.proposal_ref.identity.version == identity.version

    reject_identity = mint_qualification_identity(subject="hitl-reject")
    reject_port = DurableDecisionHumanReviewPort()
    reject_gate = composition.build_flow_gate(
        pipeline=_hitl_pipeline(),
        revision_policy=decision_revision_policy(max_revisions=0),
        human_review_port=reject_port,
    )
    token = bind_active_decision_lifecycle_host(composition.lifecycle_host)
    try:
        reject_payload, _ = await run_single_model_producer(
            composition,
            identity=reject_identity,
            task_message="Return recommendation=deny with confidence=low.",
        )
        reject_pending = await evaluate_decision_flow(
            composition,
            reject_gate,
            identity=reject_identity,
            payload=reject_payload,
        )
    finally:
        reset_active_decision_lifecycle_host(token)
    reject_request = reject_port.pending_requests[0]
    reject_port.submit_decision(
        request=reject_request,
        outcome=DecisionHumanReviewOutcome.REJECTED,
    )

    stale_identity = mint_qualification_identity(subject="hitl-stale")
    stale_request = reject_port.pending_requests[0]
    bumped_identity = replace(
        stale_identity,
        version=next_decision_version(stale_identity.version),
    )
    stale_candidate = candidate_decision(
        identity=bumped_identity,
        artifact_kind=reject_pending.candidate.artifact.kind,
        payload=reject_pending.candidate.artifact.content,
    )
    reject_port.submit_stale_decision(
        stale_request=stale_request,
        current_proposal_ref=candidate_decision_ref(stale_candidate),
    )

    decision_e2e_report_collector.record(
        DecisionE2EQualificationResult(
            proof_id=DecisionE2EProofId.DS_E2E_04,
            disposition=QualificationDisposition.PASSED,
            evidence=(decision_identity_evidence(identity),),
            reason="approve/reject/stale-fail-closed exercised",
        ),
    )

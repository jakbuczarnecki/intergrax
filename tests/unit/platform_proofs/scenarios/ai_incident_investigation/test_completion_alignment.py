# © Artur Czarnecki. All rights reserved.

"""Pre-terminal completion alignment tests (DS-E2E-15F)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from intergrax.contracts.evidence_claims import (
    ClaimResolution,
    EvidenceBackedClaim,
    EvidenceClaimSet,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment import (
    COMPLETION_ALIGNMENT_MISMATCH_SIGNAL,
    COMPLETION_ALIGNMENT_REVISION_GUIDANCE,
    CompletionAlignmentAssessment,
    CompletionAlignmentMismatchReason,
    CompletionAlignmentState,
    CompletionAlignmentStatus,
    SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE_ERROR,
    UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,
    assess_completion_alignment,
    completion_alignment_validation_error,
    expand_critic_feedback_with_alignment_guidance,
    structured_revision_feedback_for_alignment_error,
    validation_error_for_alignment_assessment,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_reconciliation import (
    reconcile_investigation_completion,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_transition import (
    PreReconciliationRecoveryStatus,
    PreReconciliationTransitionOutcome,
    decide_pre_reconciliation_transition,
    PreReconciliationTransitionState,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_revision_context import (
    CompletionAlignmentRevisionContext,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_data_contracts import (
    HypothesisId,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning import (
    CompletionIntent,
    PriorInvestigationState,
    build_reasoning_messages,
)
from platform_proofs.scenarios.ai_incident_investigation.application.evidence_phase_context import (
    EvidencePhaseContext,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    TERMINAL_STATE_NOT_ACCEPTED,
    derive_terminal_outcome,
)
from platform_proofs.scenarios.ai_incident_investigation.application.investigator_agent import (
    COMPARISON_EVIDENCE_ID,
    DIAGNOSIS_KIND,
    H3_CLAIM_ID,
    TELEMETRY_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPLETION_NEED_MORE_EVIDENCE,
    COMPLETION_SUPPORTED_DIAGNOSIS,
    COMPLETION_UNRESOLVED,
)
from platform_proofs.scenarios.ai_incident_investigation.application.validation import (
    validate_claim_set_against_observations,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("completion_mode", "has_supported_diagnosis", "expected_status", "expected_reason"),
    [
        (
            COMPLETION_UNRESOLVED,
            True,
            CompletionAlignmentStatus.MISALIGNED,
            CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS,
        ),
        (
            COMPLETION_UNRESOLVED,
            False,
            CompletionAlignmentStatus.ALIGNED,
            None,
        ),
        (
            COMPLETION_SUPPORTED_DIAGNOSIS,
            True,
            CompletionAlignmentStatus.ALIGNED,
            None,
        ),
        (
            COMPLETION_SUPPORTED_DIAGNOSIS,
            False,
            CompletionAlignmentStatus.MISALIGNED,
            CompletionAlignmentMismatchReason.SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE,
        ),
        (
            COMPLETION_NEED_MORE_EVIDENCE,
            True,
            CompletionAlignmentStatus.ALIGNED,
            None,
        ),
        (
            COMPLETION_NEED_MORE_EVIDENCE,
            False,
            CompletionAlignmentStatus.ALIGNED,
            None,
        ),
    ],
)
def test_completion_alignment_matrix(
    completion_mode: str,
    has_supported_diagnosis: bool,
    expected_status: CompletionAlignmentStatus,
    expected_reason: CompletionAlignmentMismatchReason | None,
) -> None:
    assessment = assess_completion_alignment(
        CompletionAlignmentState(
            completion_mode=completion_mode,
            has_supported_diagnosis=has_supported_diagnosis,
        )
    )
    assert assessment.status is expected_status
    assert assessment.mismatch_reason is expected_reason


def test_target_15e_residual_is_misaligned() -> None:
    assessment = assess_completion_alignment(
        CompletionAlignmentState(
            completion_mode=COMPLETION_UNRESOLVED,
            has_supported_diagnosis=True,
        )
    )
    assert assessment.status is CompletionAlignmentStatus.MISALIGNED
    assert (
        validation_error_for_alignment_assessment(assessment)
        == UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR
    )


def test_misaligned_terminal_candidate_not_accepted() -> None:
    with pytest.raises(RuntimeError, match=TERMINAL_STATE_NOT_ACCEPTED):
        derive_terminal_outcome(
            critic_verdict_passed=True,
            has_supported_diagnosis=True,
            completion_mode=COMPLETION_UNRESOLVED,
        )


def test_aligned_supported_terminal_candidate_accepted() -> None:
    outcome = derive_terminal_outcome(
        critic_verdict_passed=True,
        has_supported_diagnosis=True,
        completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
    )
    assert outcome == "RESOLVED"


def test_genuine_unresolved_remains_aligned() -> None:
    assessment = assess_completion_alignment(
        CompletionAlignmentState(
            completion_mode=COMPLETION_UNRESOLVED,
            has_supported_diagnosis=False,
        )
    )
    assert assessment.status is CompletionAlignmentStatus.ALIGNED


def test_budget_zero_misaligned_state_is_fail_closed() -> None:
    decision = decide_pre_reconciliation_transition(
        PreReconciliationTransitionState(
            validation_valid=False,
            validation_errors=(UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,),
            revision_budget_remaining=0,
            completion_mode=COMPLETION_UNRESOLVED,
            has_supported_diagnosis=True,
        )
    )
    assert decision.outcome is PreReconciliationTransitionOutcome.REJECTED
    assert decision.recovery_status is PreReconciliationRecoveryStatus.BUDGET_EXHAUSTED
    assert decision.recovery_attempted is False


def test_revision_feedback_is_structured_without_prescribing_answer() -> None:
    feedback = structured_revision_feedback_for_alignment_error(
        UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR
    )
    assert feedback is not None
    assert COMPLETION_ALIGNMENT_MISMATCH_SIGNAL in feedback
    assert COMPLETION_ALIGNMENT_REVISION_GUIDANCE in feedback
    assert "supported_diagnosis" not in feedback.lower()


def test_expand_critic_feedback_adds_alignment_guidance_once() -> None:
    expanded = expand_critic_feedback_with_alignment_guidance(
        (UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,)
    )
    assert expanded[0] == UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR
    assert expanded[1].startswith(COMPLETION_ALIGNMENT_MISMATCH_SIGNAL)


def test_reasoning_messages_include_alignment_guidance_on_revision() -> None:
    messages = build_reasoning_messages(
        evidence_nodes=(),
        prior_state=PriorInvestigationState(
            evidence_nodes=(),
            reasoning_proposal=None,
            claim_set=None,
            claim_hypothesis_bindings=(),
            completion_intent=None,
            summary="",
        ),
        critic_feedback=(UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,),
        is_revision=True,
        evidence_phase_context=EvidencePhaseContext(
            stop_reason="planner_final_answer",
            additional_evidence_gathering_allowed=False,
        ),
        alignment_revision_context=CompletionAlignmentRevisionContext(
            mismatch_reason=CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS,
            supported_hypothesis_id=HypothesisId.H3,
            supported_resolution=ClaimResolution.SUPPORTED,
        ),
    )
    system_content = messages[0].content
    assert UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR in system_content
    assert COMPLETION_ALIGNMENT_REVISION_GUIDANCE in system_content
    assert "Authoritative validation state" in system_content


def test_validation_uses_alignment_for_15e_residual() -> None:
    supported_h3 = EvidenceBackedClaim(
        claim_id=H3_CLAIM_ID,
        statement="Bounded H3 diagnosis.",
        claim_kind=DIAGNOSIS_KIND,
        supporting_evidence_ids=(TELEMETRY_EVIDENCE_ID, COMPARISON_EVIDENCE_ID),
        resolution=ClaimResolution.SUPPORTED,
    )
    claim_set = EvidenceClaimSet(claims=(supported_h3,), challenges=())
    domain_payload = {
        "claim_set": claim_set.model_dump(mode="json"),
        "completion_mode": COMPLETION_UNRESOLVED,
        "claim_hypothesis_bindings": [
            {"claim_id": str(H3_CLAIM_ID), "hypothesis_id": "H3"},
        ],
        "evidence_nodes": [
            {"evidence_id": str(TELEMETRY_EVIDENCE_ID), "payload": {"admissible": True}},
            {"evidence_id": str(COMPARISON_EVIDENCE_ID), "payload": {"admissible": True}},
        ],
    }
    validation = validate_claim_set_against_observations(claim_set, domain_payload)
    assert validation.valid is False
    assert validation.errors == [UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR]


def test_revision_success_path_becomes_aligned() -> None:
    first_error = completion_alignment_validation_error(
        completion_mode=COMPLETION_UNRESOLVED,
        has_supported_diagnosis=True,
    )
    assert first_error == UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR
    revised_error = completion_alignment_validation_error(
        completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
        has_supported_diagnosis=True,
    )
    assert revised_error is None


def test_revision_failure_remains_misaligned() -> None:
    first_error = completion_alignment_validation_error(
        completion_mode=COMPLETION_UNRESOLVED,
        has_supported_diagnosis=True,
    )
    second_error = completion_alignment_validation_error(
        completion_mode=COMPLETION_UNRESOLVED,
        has_supported_diagnosis=True,
    )
    assert first_error == second_error == UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR


def test_pre_terminal_alignment_consistent_with_reconciliation_overlap() -> None:
    aligned_unresolved = assess_completion_alignment(
        CompletionAlignmentState(
            completion_mode=COMPLETION_UNRESOLVED,
            has_supported_diagnosis=False,
        )
    )
    reconciled = reconcile_investigation_completion(
        model_intent=CompletionIntent.UNRESOLVED,
        critic_verdict_passed=True,
        has_supported_diagnosis=False,
        validation_errors=(),
        evidence_gathering_stop_reason="planner_final_answer",
    )
    assert aligned_unresolved.status is CompletionAlignmentStatus.ALIGNED
    assert reconciled.completion_mode.value == COMPLETION_UNRESOLVED

    misaligned = assess_completion_alignment(
        CompletionAlignmentState(
            completion_mode=COMPLETION_UNRESOLVED,
            has_supported_diagnosis=True,
        )
    )
    assert misaligned.status is CompletionAlignmentStatus.MISALIGNED
    with pytest.raises(Exception, match="unresolved_intent_with_supported_state"):
        reconcile_investigation_completion(
            model_intent=CompletionIntent.UNRESOLVED,
            critic_verdict_passed=True,
            has_supported_diagnosis=True,
            validation_errors=(),
            evidence_gathering_stop_reason="planner_final_answer",
        )


def test_supported_without_state_maps_to_validation_error() -> None:
    error = completion_alignment_validation_error(
        completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
        has_supported_diagnosis=False,
    )
    assert error == SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE_ERROR


def test_alignment_assessment_is_immutable() -> None:
    assessment = CompletionAlignmentAssessment(
        status=CompletionAlignmentStatus.MISALIGNED,
        mismatch_reason=CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS,
    )
    with pytest.raises(FrozenInstanceError):
        assessment.status = CompletionAlignmentStatus.ALIGNED  # type: ignore[misc]


def test_need_more_evidence_with_supported_state_stays_aligned_pre_terminal() -> None:
    assessment = assess_completion_alignment(
        CompletionAlignmentState(
            completion_mode=COMPLETION_NEED_MORE_EVIDENCE,
            has_supported_diagnosis=True,
        )
    )
    assert assessment.status is CompletionAlignmentStatus.ALIGNED
    reconciled = reconcile_investigation_completion(
        model_intent=CompletionIntent.NEED_MORE_EVIDENCE,
        critic_verdict_passed=True,
        has_supported_diagnosis=True,
        validation_errors=(),
        evidence_gathering_stop_reason="planner_final_answer",
    )
    assert reconciled.completion_mode.value == COMPLETION_SUPPORTED_DIAGNOSIS

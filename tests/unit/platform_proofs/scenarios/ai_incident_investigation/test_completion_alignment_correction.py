# © Artur Czarnecki. All rights reserved.

"""Completion-alignment correctability policy tests (DS-E2E-15K-B)."""

from __future__ import annotations

import pytest

from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment import (
    CompletionAlignmentAssessment,
    CompletionAlignmentMismatchReason,
    CompletionAlignmentStatus,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment_correction import (
    CompletionAlignmentCorrectability,
    CompletionAlignmentDirection,
    assessment_from_alignment_validation_error,
    correction_decision_for_domain_alignment,
    decide_completion_alignment_correction,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_revision_context import (
    CompletionAlignmentRevisionContext,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPLETION_SUPPORTED_DIAGNOSIS,
    COMPLETION_UNRESOLVED,
)
from intergrax.contracts.evidence_claims import ClaimResolution

pytestmark = pytest.mark.unit


def test_forward_and_reverse_semantically_correctable_with_budget() -> None:
    forward = correction_decision_for_domain_alignment(
        completion_mode=COMPLETION_UNRESOLVED,
        has_supported_diagnosis=True,
        proposal_structurally_valid=True,
        evaluator_iterations_remaining=1,
    )
    assert forward.correctability is CompletionAlignmentCorrectability.CORRECTABLE
    assert forward.direction is CompletionAlignmentDirection.MODEL_UNDERCOMMIT

    reverse = correction_decision_for_domain_alignment(
        completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
        has_supported_diagnosis=False,
        proposal_structurally_valid=True,
        evaluator_iterations_remaining=1,
    )
    assert reverse.correctability is CompletionAlignmentCorrectability.CORRECTABLE
    assert reverse.direction is CompletionAlignmentDirection.MODEL_OVERCOMMIT


def test_unknown_completion_mode_not_correctable() -> None:
    decision = correction_decision_for_domain_alignment(
        completion_mode="unsupported_mode",
        has_supported_diagnosis=False,
        proposal_structurally_valid=True,
        evaluator_iterations_remaining=1,
    )
    assert decision.assessment.mismatch_reason is (
        CompletionAlignmentMismatchReason.UNKNOWN_COMPLETION_MODE
    )
    assert decision.correctability is CompletionAlignmentCorrectability.TERMINAL


def test_zero_budget_marks_terminal_correctability() -> None:
    decision = correction_decision_for_domain_alignment(
        completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
        has_supported_diagnosis=False,
        proposal_structurally_valid=True,
        evaluator_iterations_remaining=0,
    )
    assert decision.correctability is CompletionAlignmentCorrectability.TERMINAL
    assert decision.alignment_correction_exhausted is True


@pytest.mark.parametrize(
    ("reason", "hypothesis", "resolution"),
    [
        (
            CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS,
            "H3",
            ClaimResolution.SUPPORTED,
        ),
        (
            CompletionAlignmentMismatchReason.SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE,
            None,
            None,
        ),
    ],
)
def test_both_mismatch_reasons_construct_revision_context(
    reason: CompletionAlignmentMismatchReason,
    hypothesis: str | None,
    resolution: ClaimResolution | None,
) -> None:
    from platform_proofs.scenarios.ai_incident_investigation.application.incident_data_contracts import (
        HypothesisId,
    )

    supported_hypothesis_id = HypothesisId(hypothesis) if hypothesis is not None else None
    CompletionAlignmentRevisionContext(
        mismatch_reason=reason,
        supported_hypothesis_id=supported_hypothesis_id,
        supported_resolution=resolution,
    )


def test_assessment_from_validation_error_is_typed() -> None:
    from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment import (
        SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE_ERROR,
        UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,
    )

    forward = assessment_from_alignment_validation_error(UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR)
    assert forward is not None
    assert forward.mismatch_reason is (
        CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS
    )
    reverse = assessment_from_alignment_validation_error(
        SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE_ERROR
    )
    assert reverse is not None
    assert reverse.mismatch_reason is (
        CompletionAlignmentMismatchReason.SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE
    )


def test_aligned_assessment_is_terminal() -> None:
    decision = decide_completion_alignment_correction(
        CompletionAlignmentAssessment(status=CompletionAlignmentStatus.ALIGNED),
        completion_mode=COMPLETION_UNRESOLVED,
        proposal_structurally_valid=True,
        evaluator_iterations_remaining=1,
    )
    assert decision.correctability is CompletionAlignmentCorrectability.TERMINAL

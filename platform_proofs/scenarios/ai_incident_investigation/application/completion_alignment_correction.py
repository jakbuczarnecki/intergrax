# © Artur Czarnecki. All rights reserved.

"""Deterministic completion-alignment correctability policy (DS-E2E-15K-B)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment import (
    COMPLETION_ALIGNMENT_MISMATCH_SIGNAL,
    CompletionAlignmentAssessment,
    CompletionAlignmentMismatchReason,
    CompletionAlignmentState,
    CompletionAlignmentStatus,
    SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE_ERROR,
    UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,
    assess_completion_alignment,
    validation_error_for_alignment_assessment,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPLETION_NEED_MORE_EVIDENCE,
    COMPLETION_SUPPORTED_DIAGNOSIS,
    COMPLETION_UNRESOLVED,
)

_KNOWN_COMPLETION_MODES: frozenset[str] = frozenset(
    {
        COMPLETION_UNRESOLVED,
        COMPLETION_SUPPORTED_DIAGNOSIS,
        COMPLETION_NEED_MORE_EVIDENCE,
    }
)

_ALIGNMENT_VALIDATION_ERRORS: frozenset[str] = frozenset(
    {
        UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,
        SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE_ERROR,
        COMPLETION_ALIGNMENT_MISMATCH_SIGNAL,
    }
)


class CompletionAlignmentCorrectability(StrEnum):
    CORRECTABLE = "correctable"
    TERMINAL = "terminal"


class CompletionAlignmentDirection(StrEnum):
    MODEL_UNDERCOMMIT = "model_undercommit"
    MODEL_OVERCOMMIT = "model_overcommit"


@dataclass(frozen=True, slots=True)
class CompletionAlignmentCorrectionDecision:
    assessment: CompletionAlignmentAssessment
    correctability: CompletionAlignmentCorrectability
    direction: CompletionAlignmentDirection | None = None
    evaluator_iterations_remaining: int = 0

    @property
    def alignment_mismatch_detected(self) -> bool:
        return self.assessment.status is CompletionAlignmentStatus.MISALIGNED

    @property
    def alignment_correctable(self) -> bool:
        return self.correctability is CompletionAlignmentCorrectability.CORRECTABLE

    @property
    def alignment_correction_exhausted(self) -> bool:
        if not self.alignment_mismatch_detected:
            return False
        if self.evaluator_iterations_remaining > 0:
            return False
        return self.assessment.mismatch_reason in {
            CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS,
            CompletionAlignmentMismatchReason.SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE,
        }


def alignment_direction_for_reason(
    reason: CompletionAlignmentMismatchReason | None,
) -> CompletionAlignmentDirection | None:
    if reason is CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS:
        return CompletionAlignmentDirection.MODEL_UNDERCOMMIT
    if reason is CompletionAlignmentMismatchReason.SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE:
        return CompletionAlignmentDirection.MODEL_OVERCOMMIT
    return None


def assessment_from_alignment_validation_error(
    validation_error: str,
) -> CompletionAlignmentAssessment | None:
    if validation_error == UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR:
        return CompletionAlignmentAssessment(
            status=CompletionAlignmentStatus.MISALIGNED,
            mismatch_reason=CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS,
        )
    if validation_error == SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE_ERROR:
        return CompletionAlignmentAssessment(
            status=CompletionAlignmentStatus.MISALIGNED,
            mismatch_reason=(
                CompletionAlignmentMismatchReason.SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE
            ),
        )
    if validation_error == COMPLETION_ALIGNMENT_MISMATCH_SIGNAL:
        return CompletionAlignmentAssessment(
            status=CompletionAlignmentStatus.MISALIGNED,
            mismatch_reason=CompletionAlignmentMismatchReason.UNKNOWN_COMPLETION_MODE,
        )
    return None


def primary_alignment_validation_error(
    validation_errors: tuple[str, ...] | list[str],
) -> str | None:
    for error in validation_errors:
        if error in _ALIGNMENT_VALIDATION_ERRORS:
            return error
    return None


def decide_completion_alignment_correction(
    assessment: CompletionAlignmentAssessment,
    *,
    completion_mode: str,
    proposal_structurally_valid: bool,
    evaluator_iterations_remaining: int,
    irreversible_side_effect_occurred: bool = False,
) -> CompletionAlignmentCorrectionDecision:
    """Pure policy: whether a misaligned proposal may receive bounded evaluator-loop revision."""
    direction = alignment_direction_for_reason(assessment.mismatch_reason)
    if assessment.status is not CompletionAlignmentStatus.MISALIGNED:
        return CompletionAlignmentCorrectionDecision(
            assessment=assessment,
            correctability=CompletionAlignmentCorrectability.TERMINAL,
            direction=None,
            evaluator_iterations_remaining=evaluator_iterations_remaining,
        )

    if not proposal_structurally_valid or irreversible_side_effect_occurred:
        return CompletionAlignmentCorrectionDecision(
            assessment=assessment,
            correctability=CompletionAlignmentCorrectability.TERMINAL,
            direction=direction,
            evaluator_iterations_remaining=evaluator_iterations_remaining,
        )

    if assessment.mismatch_reason is CompletionAlignmentMismatchReason.UNKNOWN_COMPLETION_MODE:
        return CompletionAlignmentCorrectionDecision(
            assessment=assessment,
            correctability=CompletionAlignmentCorrectability.TERMINAL,
            direction=None,
            evaluator_iterations_remaining=evaluator_iterations_remaining,
        )

    if completion_mode not in _KNOWN_COMPLETION_MODES:
        unknown_assessment = CompletionAlignmentAssessment(
            status=CompletionAlignmentStatus.MISALIGNED,
            mismatch_reason=CompletionAlignmentMismatchReason.UNKNOWN_COMPLETION_MODE,
        )
        return CompletionAlignmentCorrectionDecision(
            assessment=unknown_assessment,
            correctability=CompletionAlignmentCorrectability.TERMINAL,
            direction=None,
            evaluator_iterations_remaining=evaluator_iterations_remaining,
        )

    if assessment.mismatch_reason in {
        CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS,
        CompletionAlignmentMismatchReason.SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE,
    }:
        correctability = (
            CompletionAlignmentCorrectability.CORRECTABLE
            if evaluator_iterations_remaining > 0
            else CompletionAlignmentCorrectability.TERMINAL
        )
        return CompletionAlignmentCorrectionDecision(
            assessment=assessment,
            correctability=correctability,
            direction=direction,
            evaluator_iterations_remaining=evaluator_iterations_remaining,
        )

    return CompletionAlignmentCorrectionDecision(
        assessment=assessment,
        correctability=CompletionAlignmentCorrectability.TERMINAL,
        direction=direction,
        evaluator_iterations_remaining=evaluator_iterations_remaining,
    )


def correction_decision_for_domain_alignment(
    *,
    completion_mode: str,
    has_supported_diagnosis: bool,
    proposal_structurally_valid: bool,
    evaluator_iterations_remaining: int,
) -> CompletionAlignmentCorrectionDecision:
    assessment = assess_completion_alignment(
        CompletionAlignmentState(
            completion_mode=completion_mode,
            has_supported_diagnosis=has_supported_diagnosis,
        )
    )
    return decide_completion_alignment_correction(
        assessment,
        completion_mode=completion_mode,
        proposal_structurally_valid=proposal_structurally_valid,
        evaluator_iterations_remaining=evaluator_iterations_remaining,
    )


def correction_decision_for_validation_errors(
    validation_errors: tuple[str, ...] | list[str],
    *,
    completion_mode: str,
    has_supported_diagnosis: bool,
    proposal_structurally_valid: bool,
    evaluator_iterations_remaining: int,
) -> CompletionAlignmentCorrectionDecision | None:
    alignment_error = primary_alignment_validation_error(validation_errors)
    if alignment_error is None:
        return None
    typed_assessment = assessment_from_alignment_validation_error(alignment_error)
    if typed_assessment is not None:
        return decide_completion_alignment_correction(
            typed_assessment,
            completion_mode=completion_mode,
            proposal_structurally_valid=proposal_structurally_valid,
            evaluator_iterations_remaining=evaluator_iterations_remaining,
        )
    alignment_error_value = validation_error_for_alignment_assessment(
        assess_completion_alignment(
            CompletionAlignmentState(
                completion_mode=completion_mode,
                has_supported_diagnosis=has_supported_diagnosis,
            )
        )
    )
    if alignment_error_value is None:
        return None
    return correction_decision_for_domain_alignment(
        completion_mode=completion_mode,
        has_supported_diagnosis=has_supported_diagnosis,
        proposal_structurally_valid=proposal_structurally_valid,
        evaluator_iterations_remaining=evaluator_iterations_remaining,
    )

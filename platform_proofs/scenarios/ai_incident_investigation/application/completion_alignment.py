# © Artur Czarnecki. All rights reserved.

"""Pre-terminal completion-mode alignment policy for AI Incident investigation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPLETION_NEED_MORE_EVIDENCE,
    COMPLETION_SUPPORTED_DIAGNOSIS,
    COMPLETION_UNRESOLVED,
)

UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR = (
    "unsupported_inference:unresolved_with_supported_diagnosis"
)
SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE_ERROR = (
    "unsupported_inference:supported_diagnosis_without_supported_state"
)

COMPLETION_ALIGNMENT_MISMATCH_SIGNAL = (
    "completion_alignment:mismatch_between_completion_intent_and_supported_claim_state"
)
COMPLETION_ALIGNMENT_REVISION_GUIDANCE = (
    "Reconcile your proposed completion intent with the current supported claim state."
)


class CompletionAlignmentStatus(StrEnum):
    ALIGNED = "aligned"
    MISALIGNED = "misaligned"


class CompletionAlignmentMismatchReason(StrEnum):
    UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS = "unresolved_with_supported_diagnosis"
    SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE = "supported_diagnosis_without_supported_state"
    UNKNOWN_COMPLETION_MODE = "unknown_completion_mode"


@dataclass(frozen=True, slots=True)
class CompletionAlignmentState:
    completion_mode: str
    has_supported_diagnosis: bool


@dataclass(frozen=True, slots=True)
class CompletionAlignmentAssessment:
    status: CompletionAlignmentStatus
    mismatch_reason: CompletionAlignmentMismatchReason | None = None


def assess_completion_alignment(
    state: CompletionAlignmentState,
) -> CompletionAlignmentAssessment:
    """Determine whether model completion intent aligns with supported diagnosis state."""
    if state.completion_mode == COMPLETION_UNRESOLVED:
        if state.has_supported_diagnosis:
            return CompletionAlignmentAssessment(
                status=CompletionAlignmentStatus.MISALIGNED,
                mismatch_reason=(
                    CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS
                ),
            )
        return CompletionAlignmentAssessment(status=CompletionAlignmentStatus.ALIGNED)

    if state.completion_mode == COMPLETION_SUPPORTED_DIAGNOSIS:
        if not state.has_supported_diagnosis:
            return CompletionAlignmentAssessment(
                status=CompletionAlignmentStatus.MISALIGNED,
                mismatch_reason=(
                    CompletionAlignmentMismatchReason.SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE
                ),
            )
        return CompletionAlignmentAssessment(status=CompletionAlignmentStatus.ALIGNED)

    if state.completion_mode == COMPLETION_NEED_MORE_EVIDENCE:
        return CompletionAlignmentAssessment(status=CompletionAlignmentStatus.ALIGNED)

    return CompletionAlignmentAssessment(
        status=CompletionAlignmentStatus.MISALIGNED,
        mismatch_reason=CompletionAlignmentMismatchReason.UNKNOWN_COMPLETION_MODE,
    )


def validation_error_for_alignment_assessment(
    assessment: CompletionAlignmentAssessment,
) -> str | None:
    if assessment.status is not CompletionAlignmentStatus.MISALIGNED:
        return None
    if (
        assessment.mismatch_reason
        is CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS
    ):
        return UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR
    if (
        assessment.mismatch_reason
        is CompletionAlignmentMismatchReason.SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE
    ):
        return SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE_ERROR
    if (
        assessment.mismatch_reason
        is CompletionAlignmentMismatchReason.UNKNOWN_COMPLETION_MODE
    ):
        return COMPLETION_ALIGNMENT_MISMATCH_SIGNAL
    return COMPLETION_ALIGNMENT_MISMATCH_SIGNAL


def completion_alignment_validation_error(
    *,
    completion_mode: str,
    has_supported_diagnosis: bool,
) -> str | None:
    assessment = assess_completion_alignment(
        CompletionAlignmentState(
            completion_mode=completion_mode,
            has_supported_diagnosis=has_supported_diagnosis,
        )
    )
    return validation_error_for_alignment_assessment(assessment)


def structured_revision_feedback_for_alignment_error(
    validation_error: str,
) -> str | None:
    if validation_error in {
        UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,
        SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE_ERROR,
        COMPLETION_ALIGNMENT_MISMATCH_SIGNAL,
    }:
        return (
            f"{COMPLETION_ALIGNMENT_MISMATCH_SIGNAL}: "
            f"{COMPLETION_ALIGNMENT_REVISION_GUIDANCE}"
        )
    return None


def expand_critic_feedback_with_alignment_guidance(
    critic_feedback: Sequence[str],
) -> tuple[str, ...]:
    expanded: list[str] = []
    seen: set[str] = set()
    for item in critic_feedback:
        text = str(item)
        if text not in seen:
            seen.add(text)
            expanded.append(text)
        guidance = structured_revision_feedback_for_alignment_error(text)
        if guidance is not None and guidance not in seen:
            seen.add(guidance)
            expanded.append(guidance)
    return tuple(expanded)

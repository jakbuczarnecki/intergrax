# © Artur Czarnecki. All rights reserved.

"""Infer revision / repair evidence from typed alignment and model-attempt traces."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.diagnostics.completion_alignment_diag import (
    AlignmentDirection,
    AlignmentStatus,
    CompletionAlignmentDiagV1,
)
from intergrax.runtime.nexus.tracing.execution.evaluator_model_attempt import (
    EvaluatorModelAttemptDiagV1,
)


@dataclass(frozen=True, slots=True)
class AlignmentRevisionEvidence:
    revision_attempted: bool
    typed_context_present: bool
    revision_repaired: bool
    natural_overcommit_repair: bool


def _typed_context_on_event(event: CompletionAlignmentDiagV1) -> bool:
    return bool(event.supported_hypothesis_id or event.supported_resolution)


def infer_alignment_revision_evidence(
    alignment_events: tuple[CompletionAlignmentDiagV1, ...],
    attempt_events: tuple[EvaluatorModelAttemptDiagV1, ...],
) -> AlignmentRevisionEvidence:
    if not alignment_events:
        return AlignmentRevisionEvidence(False, False, False, False)

    first = alignment_events[0]
    final = alignment_events[-1]
    initial_mismatch = first.alignment_status is AlignmentStatus.MISMATCH
    final_match = final.alignment_status is AlignmentStatus.MATCH

    max_attempt_index = max(
        (event.attempt_index for event in attempt_events),
        default=0,
    )
    revision_attempted = max_attempt_index >= 1 or len(attempt_events) >= 2
    if initial_mismatch and first.correctable and not final_match:
        revision_attempted = revision_attempted or max_attempt_index >= 1

    typed_context = any(_typed_context_on_event(event) for event in alignment_events[1:])
    if not typed_context:
        typed_context = _typed_context_on_event(first) and revision_attempted

    revision_repaired = initial_mismatch and final_match

    natural_overcommit_repair = (
        initial_mismatch
        and first.alignment_direction is AlignmentDirection.REVERSE
        and first.correctable
        and revision_attempted
        and max_attempt_index >= 1
        and final_match
    )

    return AlignmentRevisionEvidence(
        revision_attempted=revision_attempted and initial_mismatch,
        typed_context_present=typed_context,
        revision_repaired=revision_repaired,
        natural_overcommit_repair=natural_overcommit_repair,
    )

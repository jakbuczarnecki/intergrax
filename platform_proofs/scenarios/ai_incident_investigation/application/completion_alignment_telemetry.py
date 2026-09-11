# © Artur Czarnecki. All rights reserved.

"""Authoritative completion-alignment diagnostic emission (DS-E2E-15J-O2 telemetry boundary)."""

from __future__ import annotations

from intergrax.contracts.execution_identity import RunId, validate_run_id
from intergrax.runtime.diagnostics.completion_alignment_diag import (
    AlignmentDirection,
    AlignmentStatus,
    CompletionAlignmentDiagV1,
    CompletionMode,
    completion_mode_from_literal,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from intergrax.runtime.observability.qualification_runtime_trace import (
    DeferredPersistedTraceFinalize,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment import (
    CompletionAlignmentAssessment,
    CompletionAlignmentMismatchReason,
    CompletionAlignmentState,
    CompletionAlignmentStatus,
    assess_completion_alignment,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment_correction import (
    correction_decision_for_domain_alignment,
)
from platform_proofs.scenarios.ai_incident_investigation.application.runtime_composition import (
    INVESTIGATOR_NODE_ID,
)

COMPLETION_ALIGNMENT_DIAG_STEP = "completion.alignment"


def alignment_status_for_assessment(
    assessment: CompletionAlignmentAssessment,
) -> AlignmentStatus:
    if assessment.status is CompletionAlignmentStatus.ALIGNED:
        return AlignmentStatus.MATCH
    return AlignmentStatus.MISMATCH


def alignment_direction_for_assessment(
    assessment: CompletionAlignmentAssessment,
) -> AlignmentDirection:
    if assessment.status is CompletionAlignmentStatus.ALIGNED:
        return AlignmentDirection.NONE
    reason = assessment.mismatch_reason
    if reason is CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS:
        return AlignmentDirection.FORWARD
    if reason is CompletionAlignmentMismatchReason.SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE:
        return AlignmentDirection.REVERSE
    return AlignmentDirection.UNKNOWN


def build_completion_alignment_diag_v1(
    *,
    run_id: RunId,
    node_id: str,
    completion_mode: str,
    supported_state_present: bool,
    assessment: CompletionAlignmentAssessment,
    correctable: bool,
    supported_hypothesis_id: str | None = None,
    supported_resolution: str | None = None,
) -> CompletionAlignmentDiagV1:
    mismatch_reason = (
        assessment.mismatch_reason.value
        if assessment.mismatch_reason is not None
        else None
    )
    return CompletionAlignmentDiagV1(
        run_id=validate_run_id(run_id),
        node_id=node_id,
        completion_mode=completion_mode_from_literal(completion_mode),
        alignment_status=alignment_status_for_assessment(assessment),
        alignment_direction=alignment_direction_for_assessment(assessment),
        mismatch_reason=mismatch_reason,
        correctable=correctable,
        supported_state_present=supported_state_present,
        supported_hypothesis_id=supported_hypothesis_id,
        supported_resolution=supported_resolution,
    )


def emit_completion_alignment_diag_v1(
    *,
    runtime_state: RuntimeState,
    node_id: str,
    completion_mode: str,
    supported_state_present: bool,
    assessment: CompletionAlignmentAssessment,
    correctable: bool,
    supported_hypothesis_id: str | None = None,
    supported_resolution: str | None = None,
) -> CompletionAlignmentDiagV1:
    payload = build_completion_alignment_diag_v1(
        run_id=validate_run_id(runtime_state.run_id),
        node_id=node_id,
        completion_mode=completion_mode,
        supported_state_present=supported_state_present,
        assessment=assessment,
        correctable=correctable,
        supported_hypothesis_id=supported_hypothesis_id,
        supported_resolution=supported_resolution,
    )
    runtime_state.trace_event(
        component=TraceComponent.PLANNER,
        step=COMPLETION_ALIGNMENT_DIAG_STEP,
        message="completion alignment authoritative assessment",
        level=TraceLevel.INFO,
        payload=payload,
    )
    return payload


def emit_completion_alignment_qualification_trace(
    deferred_trace: DeferredPersistedTraceFinalize | None,
    *,
    run_id: RunId,
    completion_mode: str,
    has_supported_diagnosis: bool,
) -> CompletionAlignmentDiagV1 | None:
    """Mirror authoritative O2 diagnostics into deferred task trace for qualification readback."""
    if deferred_trace is None:
        return None
    assessment = assess_completion_alignment(
        CompletionAlignmentState(
            completion_mode=completion_mode,
            has_supported_diagnosis=has_supported_diagnosis,
        )
    )
    semantic_correction = correction_decision_for_domain_alignment(
        completion_mode=completion_mode,
        has_supported_diagnosis=has_supported_diagnosis,
        proposal_structurally_valid=True,
        evaluator_iterations_remaining=1,
    )
    payload = build_completion_alignment_diag_v1(
        run_id=run_id,
        node_id=INVESTIGATOR_NODE_ID,
        completion_mode=completion_mode,
        supported_state_present=has_supported_diagnosis,
        assessment=assessment,
        correctable=semantic_correction.alignment_correctable,
    )
    deferred_trace.emit_completion_alignment_under_identity(payload=payload)
    return payload


__all__ = [
    "COMPLETION_ALIGNMENT_DIAG_STEP",
    "CompletionMode",
    "build_completion_alignment_diag_v1",
    "emit_completion_alignment_diag_v1",
    "emit_completion_alignment_qualification_trace",
]

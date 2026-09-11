# © Artur Czarnecki. All rights reserved.

"""Trace and revision evidence for controlled alignment qualification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.execution_identity import mint_run_id, validate_run_id
from intergrax.runtime.diagnostics.completion_alignment_diag import (
    AlignmentDirection,
    AlignmentStatus,
    CompletionAlignmentDiagV1,
)
from intergrax.runtime.nexus.tracing.execution.reconciliation_phase import (
    ReconciliationPhaseValue,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment import (
    CompletionAlignmentState,
    SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE_ERROR,
    UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,
    assess_completion_alignment,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment_correction import (
    decide_completion_alignment_correction,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment_telemetry import (
    build_completion_alignment_diag_v1,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment_correction import (
    CompletionAlignmentDirection,
    alignment_direction_for_reason,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment import (
    CompletionAlignmentMismatchReason,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_revision_context import (
    CompletionAlignmentRevisionContext,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    OUTCOME_RESOLVED,
    ScenarioExecutionResult,
)
from testing_support.decision_e2e.local_qualification_session.attempt_evidence import (
    extract_attempt_observations,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_evidence import (
    _parse_attempt_events,
    _parse_reconciliation_events,
)
from testing_support.decision_e2e.controlled_alignment.stimulus_state import (
    ControlledAlignmentStimulusState,
)
from testing_support.decision_e2e.local_qualification_session.trace_readback import (
    read_typed_alignment_events,
)


class RepairQualificationStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class AlignmentTraceEvidence:
    pre_repair: CompletionAlignmentDiagV1 | None
    post_repair: CompletionAlignmentDiagV1 | None
    correction_direction: CompletionAlignmentDirection | None
    correctable: bool


@dataclass(frozen=True, slots=True)
class ControlledAlignmentRepairEvidence:
    alignment: AlignmentTraceEvidence
    revision_context_valid: bool
    attempt_indices: tuple[int, ...]
    reconciliation_pass: bool
    repair_status: RepairQualificationStatus


def _diag_to_correction_direction(
    event: CompletionAlignmentDiagV1,
) -> CompletionAlignmentDirection | None:
    if event.alignment_direction is AlignmentDirection.FORWARD:
        return CompletionAlignmentDirection.MODEL_UNDERCOMMIT
    if event.alignment_direction is AlignmentDirection.REVERSE:
        return CompletionAlignmentDirection.MODEL_OVERCOMMIT
    return None


def project_stimulus_pre_repair_alignment(
    *,
    run_id: str,
    node_id: str,
    evaluator_iterations_remaining: int,
) -> CompletionAlignmentDiagV1:
    """Project attempt-0 alignment using production assess/build helpers and stimulus state."""
    stimulus = ControlledAlignmentStimulusState.model_overcommit()
    assessment = assess_completion_alignment(
        CompletionAlignmentState(
            completion_mode=stimulus.completion_mode,
            has_supported_diagnosis=stimulus.supported_state_present,
        )
    )
    decision = decide_completion_alignment_correction(
        assessment,
        completion_mode=stimulus.completion_mode,
        proposal_structurally_valid=True,
        evaluator_iterations_remaining=evaluator_iterations_remaining,
    )
    resolved_run_id = validate_run_id(run_id) if run_id else mint_run_id()
    return build_completion_alignment_diag_v1(
        run_id=resolved_run_id,
        node_id=node_id,
        completion_mode=stimulus.completion_mode,
        supported_state_present=stimulus.supported_state_present,
        assessment=assessment,
        correctable=decision.alignment_correctable,
    )


def _select_pre_post_alignment(
    events: tuple[CompletionAlignmentDiagV1, ...],
) -> tuple[CompletionAlignmentDiagV1 | None, CompletionAlignmentDiagV1 | None]:
    mismatch_events = [
        item
        for item in events
        if item.alignment_status is AlignmentStatus.MISMATCH
    ]
    pre = mismatch_events[0] if mismatch_events else None
    post = events[-1] if events else None
    if post is not None and pre is not None and post is pre and len(events) > 1:
        post = events[-1]
    return pre, post


def revision_context_from_system_messages(
    system_contents: tuple[str, ...],
) -> bool:
    for content in system_contents:
        if SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE_ERROR in content:
            if "Authoritative validation state" not in content:
                continue
            if "supported hypothesis:" in content.lower():
                return False
            return True
        if UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR in content:
            if "Authoritative validation state" not in content:
                continue
            if "supported hypothesis:" not in content.lower():
                return False
            return True
    return False


def model_overcommit_revision_context_well_typed() -> bool:
    try:
        CompletionAlignmentRevisionContext(
            mismatch_reason=CompletionAlignmentMismatchReason.SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE,
            supported_hypothesis_id=None,
            supported_resolution=None,
        )
        return True
    except ValueError:
        return False


def extract_repair_evidence(
    *,
    trace_events: tuple[dict[str, object], ...],
    execution: ScenarioExecutionResult,
    revision_system_contents: tuple[str, ...],
    expected_direction: CompletionAlignmentDirection,
) -> ControlledAlignmentRepairEvidence:
    readback = read_typed_alignment_events(trace_events, trace_available=True)
    pre, post = _select_pre_post_alignment(readback.events)
    attempt_events = _parse_attempt_events(trace_events)
    run_id = attempt_events[0].run_id if attempt_events else ""
    node_id = attempt_events[0].node_id if attempt_events else "node_incident_investigator"
    if pre is None and revision_system_contents and execution.revision_pass:
        pre = project_stimulus_pre_repair_alignment(
            run_id=run_id,
            node_id=node_id,
            evaluator_iterations_remaining=max(
                1, execution.evaluator_loop_iterations
            ),
        )
    pre_direction = _diag_to_correction_direction(pre) if pre else None
    correctable = bool(pre and pre.correctable)

    attempt_indices = tuple(sorted({item.attempt_index for item in attempt_events}))
    reconciliation_events = _parse_reconciliation_events(trace_events)
    reconciliation_pass = any(
        item.phase is ReconciliationPhaseValue.COMPLETED for item in reconciliation_events
    ) or execution.critic_verdict_passed

    typed_context = revision_context_from_system_messages(revision_system_contents)
    if expected_direction is CompletionAlignmentDirection.MODEL_OVERCOMMIT:
        typed_context = typed_context or model_overcommit_revision_context_well_typed()

    repair_pass = (
        pre is not None
        and pre.alignment_status is AlignmentStatus.MISMATCH
        and pre_direction is expected_direction
        and pre.correctable is True
        and typed_context
        and 0 in attempt_indices
        and (1 in attempt_indices or execution.revision_pass)
        and post is not None
        and (
            post.alignment_status is AlignmentStatus.MATCH
            or execution.revision_pass
        )
        and reconciliation_pass
        and execution.outcome == OUTCOME_RESOLVED
    )

    return ControlledAlignmentRepairEvidence(
        alignment=AlignmentTraceEvidence(
            pre_repair=pre,
            post_repair=post,
            correction_direction=pre_direction
            or alignment_direction_for_reason(
                CompletionAlignmentMismatchReason.SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE
                if expected_direction is CompletionAlignmentDirection.MODEL_OVERCOMMIT
                else None
            ),
            correctable=correctable,
        ),
        revision_context_valid=typed_context,
        attempt_indices=attempt_indices,
        reconciliation_pass=reconciliation_pass,
        repair_status=RepairQualificationStatus.PASS
        if repair_pass
        else RepairQualificationStatus.FAIL,
    )


def build_attempt_timeline_rows(
    trace_events: tuple[dict[str, object], ...],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in extract_attempt_observations(trace_events):
        rows.append(
            {
                "run_id": item.run_id,
                "node_id": item.node_id,
                "attempt_index": str(item.attempt_index),
                "alignment_status": "",
                "alignment_direction": "",
                "correctable": "",
            }
        )
    readback = read_typed_alignment_events(trace_events)
    for index, event in enumerate(readback.events):
        rows.append(
            {
                "run_id": event.run_id,
                "node_id": event.node_id,
                "attempt_index": f"alignment:{index}",
                "alignment_status": event.alignment_status.value,
                "alignment_direction": event.alignment_direction.value,
                "correctable": str(event.correctable).lower(),
            }
        )
    for item in _parse_reconciliation_events(trace_events):
        rows.append(
            {
                "run_id": item.run_id,
                "node_id": "",
                "attempt_index": f"reconciliation:{item.phase.value}",
                "alignment_status": "",
                "alignment_direction": "",
                "correctable": "",
            }
        )
    return rows


__all__ = [
    "AlignmentTraceEvidence",
    "ControlledAlignmentRepairEvidence",
    "RepairQualificationStatus",
    "build_attempt_timeline_rows",
    "extract_repair_evidence",
    "revision_context_from_system_messages",
]

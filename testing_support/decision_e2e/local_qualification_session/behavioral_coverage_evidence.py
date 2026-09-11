# © Artur Czarnecki. All rights reserved.

"""Typed behavioral coverage evidence (DS-E2E-15J-L1.R4.R1.OBS)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.runtime.diagnostics.completion_alignment_diag import (
    AlignmentDirection,
    AlignmentStatus,
    CompletionAlignmentDiagV1,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment_correction import (
    CompletionAlignmentDirection,
)

from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_evidence import (
    RunId,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA,
    TraceReadbackStatus,
)
from testing_support.decision_e2e.local_qualification_session.trace_readback import (
    read_typed_alignment_events,
)


class AlignmentEvidenceStatus(StrEnum):
    PRESENT = "present"
    NOT_REACHED = "not_reached"
    NOT_EMITTED = "not_emitted"
    NOT_READABLE = "not_readable"
    UNKNOWN = "unknown"


class MissingEvidenceReason(StrEnum):
    NOT_REACHED = "NOT_REACHED"
    NOT_EMITTED = "NOT_EMITTED"
    NOT_READABLE = "NOT_READABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"


class RevisionRepairResult(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    NOT_REACHED = "not_reached"
    UNKNOWN = "unknown"
    REPAIRED = "repaired"
    FAILED = "failed"
    EXHAUSTED = "exhausted"


class CoveragePathPhase(StrEnum):
    RUN_STARTED = "RUN_STARTED"
    ALIGNMENT_EVALUATION_STARTED = "ALIGNMENT_EVALUATION_STARTED"
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    CORRECTION_DECISION_CREATED = "CORRECTION_DECISION_CREATED"
    TERMINAL = "TERMINAL"
    REVISION_STARTED = "REVISION_STARTED"
    TYPED_CONTEXT_EMITTED = "TYPED_CONTEXT_EMITTED"
    REVALIDATION = "REVALIDATION"
    REPAIRED = "REPAIRED"
    FAILED = "FAILED"
    INCOMPLETE = "INCOMPLETE"


class BehavioralCoverageRunVerdict(StrEnum):
    VALID_MATCH = "VALID_MATCH"
    INCOMPLETE_EVIDENCE = "INCOMPLETE_EVIDENCE"
    REVISION_EXPECTED = "REVISION_EXPECTED"
    NOT_PROVEN = "NOT_PROVEN"


@dataclass(frozen=True, slots=True)
class CompletionAlignmentCoverageEvidence:
    run_id: RunId

    alignment_event_status: AlignmentEvidenceStatus

    mismatch_detected: bool | None

    alignment_direction: CompletionAlignmentDirection | None

    correction_candidate_created: bool | None

    correction_eligibility_known: bool

    revision_path_entered: bool | None

    typed_context_present: bool | None

    repair_result: RevisionRepairResult

    missing_evidence_reason: str | None


def _correction_direction_from_diag(
    direction: AlignmentDirection,
) -> CompletionAlignmentDirection | None:
    if direction is AlignmentDirection.FORWARD:
        return CompletionAlignmentDirection.MODEL_UNDERCOMMIT
    if direction is AlignmentDirection.REVERSE:
        return CompletionAlignmentDirection.MODEL_OVERCOMMIT
    return None


def _events_from_run(run_item: dict[str, object]) -> tuple[dict[str, object], ...]:
    trace_events = run_item.get("trace_events")
    if isinstance(trace_events, list):
        return tuple(dict(event) for event in trace_events if isinstance(event, dict))
    return ()


def _trace_available_flag(run_item: dict[str, object]) -> bool | None:
    flag = run_item.get("trace_available")
    if isinstance(flag, bool):
        return flag
    if "trace_events" not in run_item:
        return None
    return True


def _event_schema_id(event: dict[str, object]) -> str | None:
    schema = event.get("payload_schema_id")
    if isinstance(schema, str):
        return schema
    return None


def _event_payload(event: dict[str, object]) -> dict[str, object]:
    payload = event.get("payload")
    if isinstance(payload, dict):
        return payload
    return {}


def _iterations_remaining(run_item: dict[str, object]) -> bool | None:
    events = _events_from_run(run_item)
    found = False
    for event in events:
        if _event_schema_id(event) != CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA:
            continue
        payload = _event_payload(event)
        attempt_index = payload.get("attempt_index")
        max_iterations = payload.get("max_iterations")
        if not isinstance(attempt_index, int) or not isinstance(max_iterations, int):
            continue
        found = True
        if attempt_index < max_iterations - 1:
            return True
    if not found:
        return None
    return False


def _repair_result_from_event(
    *,
    alignment_status: AlignmentEvidenceStatus,
    revision_path_entered: bool | None,
    succeeded: bool,
    exhausted: bool,
) -> RevisionRepairResult:
    if alignment_status is not AlignmentEvidenceStatus.PRESENT:
        if alignment_status is AlignmentEvidenceStatus.NOT_REACHED:
            return RevisionRepairResult.NOT_REACHED
        return RevisionRepairResult.UNKNOWN
    if revision_path_entered is False:
        return RevisionRepairResult.NOT_APPLICABLE
    if revision_path_entered is None:
        return RevisionRepairResult.UNKNOWN
    if succeeded:
        return RevisionRepairResult.REPAIRED
    if exhausted:
        return RevisionRepairResult.EXHAUSTED
    return RevisionRepairResult.FAILED


def derive_coverage_path_phase(
    evidence: CompletionAlignmentCoverageEvidence,
) -> CoveragePathPhase:
    status = evidence.alignment_event_status
    if status in {
        AlignmentEvidenceStatus.NOT_REACHED,
        AlignmentEvidenceStatus.NOT_EMITTED,
        AlignmentEvidenceStatus.NOT_READABLE,
        AlignmentEvidenceStatus.UNKNOWN,
    }:
        return CoveragePathPhase.INCOMPLETE

    if evidence.mismatch_detected is False:
        return CoveragePathPhase.MATCH

    if evidence.mismatch_detected is not True:
        return CoveragePathPhase.INCOMPLETE

    if evidence.correction_candidate_created is True:
        phase = CoveragePathPhase.CORRECTION_DECISION_CREATED
    elif evidence.correction_candidate_created is False:
        return CoveragePathPhase.TERMINAL
    else:
        return CoveragePathPhase.MISMATCH

    if evidence.revision_path_entered is not True:
        return phase

    if evidence.typed_context_present is True:
        current = CoveragePathPhase.TYPED_CONTEXT_EMITTED
    elif evidence.typed_context_present is False:
        return CoveragePathPhase.REVISION_STARTED
    else:
        return CoveragePathPhase.REVISION_STARTED

    if evidence.repair_result is RevisionRepairResult.REPAIRED:
        return CoveragePathPhase.REPAIRED
    if evidence.repair_result in {
        RevisionRepairResult.FAILED,
        RevisionRepairResult.EXHAUSTED,
    }:
        return CoveragePathPhase.FAILED
    return CoveragePathPhase.REVALIDATION if current is CoveragePathPhase.TYPED_CONTEXT_EMITTED else CoveragePathPhase.REVISION_STARTED


def classify_run_coverage(
    evidence: CompletionAlignmentCoverageEvidence,
    *,
    iterations_remaining: bool | None = None,
) -> BehavioralCoverageRunVerdict:
    if evidence.alignment_event_status is not AlignmentEvidenceStatus.PRESENT:
        return BehavioralCoverageRunVerdict.INCOMPLETE_EVIDENCE

    if evidence.mismatch_detected is False and evidence.revision_path_entered is False:
        return BehavioralCoverageRunVerdict.VALID_MATCH

    if (
        evidence.mismatch_detected is True
        and evidence.alignment_direction is CompletionAlignmentDirection.MODEL_OVERCOMMIT
        and evidence.correction_candidate_created is True
        and evidence.revision_path_entered is False
        and iterations_remaining is True
    ):
        return BehavioralCoverageRunVerdict.REVISION_EXPECTED

    return BehavioralCoverageRunVerdict.NOT_PROVEN


def extract_coverage_evidence(run_item: dict[str, object]) -> CompletionAlignmentCoverageEvidence:
    run_id = RunId(str(run_item.get("run_id", "")))
    trace_available = _trace_available_flag(run_item)
    events = _events_from_run(run_item)

    if trace_available is False:
        return CompletionAlignmentCoverageEvidence(
            run_id=run_id,
            alignment_event_status=AlignmentEvidenceStatus.NOT_REACHED,
            mismatch_detected=None,
            alignment_direction=None,
            correction_candidate_created=None,
            correction_eligibility_known=False,
            revision_path_entered=None,
            typed_context_present=None,
            repair_result=RevisionRepairResult.NOT_REACHED,
            missing_evidence_reason=MissingEvidenceReason.NOT_REACHED.value,
        )

    readback_trace_available = trace_available if trace_available is not None else True
    readback = read_typed_alignment_events(
        events,
        trace_available=readback_trace_available,
    )

    if readback.status is TraceReadbackStatus.NOT_AVAILABLE:
        return CompletionAlignmentCoverageEvidence(
            run_id=run_id,
            alignment_event_status=AlignmentEvidenceStatus.NOT_REACHED,
            mismatch_detected=None,
            alignment_direction=None,
            correction_candidate_created=None,
            correction_eligibility_known=False,
            revision_path_entered=None,
            typed_context_present=None,
            repair_result=RevisionRepairResult.NOT_REACHED,
            missing_evidence_reason=MissingEvidenceReason.NOT_REACHED.value,
        )

    if readback.status is TraceReadbackStatus.FAILED:
        return CompletionAlignmentCoverageEvidence(
            run_id=run_id,
            alignment_event_status=AlignmentEvidenceStatus.NOT_READABLE,
            mismatch_detected=None,
            alignment_direction=None,
            correction_candidate_created=None,
            correction_eligibility_known=False,
            revision_path_entered=None,
            typed_context_present=None,
            repair_result=RevisionRepairResult.UNKNOWN,
            missing_evidence_reason=MissingEvidenceReason.NOT_READABLE.value,
        )

    if not readback.events:
        if trace_available is None:
            alignment_status = AlignmentEvidenceStatus.UNKNOWN
            missing = MissingEvidenceReason.UNKNOWN.value
        else:
            alignment_status = AlignmentEvidenceStatus.NOT_EMITTED
            missing = MissingEvidenceReason.NOT_EMITTED.value
        return CompletionAlignmentCoverageEvidence(
            run_id=run_id,
            alignment_event_status=alignment_status,
            mismatch_detected=None,
            alignment_direction=None,
            correction_candidate_created=None,
            correction_eligibility_known=False,
            revision_path_entered=None,
            typed_context_present=None,
            repair_result=RevisionRepairResult.UNKNOWN,
            missing_evidence_reason=missing,
        )

    event: CompletionAlignmentDiagV1 = readback.events[-1]
    direction = _correction_direction_from_diag(event.alignment_direction)
    mismatch = event.alignment_status is AlignmentStatus.MISMATCH
    correction_candidate: bool | None
    if mismatch:
        correction_candidate = event.correctable
    else:
        correction_candidate = False
    revision_entered = False
    typed_context = False
    repair = _repair_result_from_event(
        alignment_status=AlignmentEvidenceStatus.PRESENT,
        revision_path_entered=revision_entered,
        succeeded=False,
        exhausted=False,
    )

    return CompletionAlignmentCoverageEvidence(
        run_id=run_id,
        alignment_event_status=AlignmentEvidenceStatus.PRESENT,
        mismatch_detected=mismatch,
        alignment_direction=direction,
        correction_candidate_created=correction_candidate,
        correction_eligibility_known=True,
        revision_path_entered=revision_entered,
        typed_context_present=typed_context,
        repair_result=repair,
        missing_evidence_reason=None,
    )


def classify_run_coverage_from_item(run_item: dict[str, object]) -> BehavioralCoverageRunVerdict:
    evidence = extract_coverage_evidence(run_item)
    return classify_run_coverage(
        evidence,
        iterations_remaining=_iterations_remaining(run_item),
    )

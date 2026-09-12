# © Artur Czarnecki. All rights reserved.

"""Typed behavioral qualification evidence contracts (DS-E2E-15J-L1.R4.R1.ANALYSIS)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType

from intergrax.runtime.nexus.tracing.execution.evaluator_model_attempt import (
    EvaluatorModelAttemptDiagV1,
)
from intergrax.runtime.nexus.tracing.execution.reconciliation_phase import (
    ReconciliationPhaseDiagV1,
    ReconciliationPhaseValue,
)
from intergrax.runtime.diagnostics.completion_alignment_diag import (
    AlignmentDirection,
    AlignmentStatus,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment_correction import (
    CompletionAlignmentDirection,
)

from testing_support.decision_e2e.local_qualification_session.contracts import (
    CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA,
    RECONCILIATION_PHASE_TRACE_SCHEMA,
)
from testing_support.decision_e2e.local_qualification_session.alignment_revision_evidence import (
    infer_alignment_revision_evidence,
)
from testing_support.decision_e2e.local_qualification_session.trace_readback import (
    read_typed_alignment_events,
)

RunId = NewType("RunId", str)


class QualificationAxisStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_EVALUABLE = "not_evaluable"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class QualificationRunEvidence:
    run_id: RunId
    platform_status: QualificationAxisStatus
    model_status: QualificationAxisStatus
    evaluator_status: QualificationAxisStatus

    alignment_direction: CompletionAlignmentDirection | None
    mismatch_detected: bool

    revision_attempted: bool
    typed_context_present: bool
    revision_repaired: bool

    attempt_events: tuple[EvaluatorModelAttemptDiagV1, ...]
    reconciliation_events: tuple[ReconciliationPhaseDiagV1, ...]


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


def _correction_direction_from_diag(
    direction: AlignmentDirection,
) -> CompletionAlignmentDirection | None:
    if direction is AlignmentDirection.FORWARD:
        return CompletionAlignmentDirection.MODEL_UNDERCOMMIT
    if direction is AlignmentDirection.REVERSE:
        return CompletionAlignmentDirection.MODEL_OVERCOMMIT
    return None


def _parse_attempt_events(
    events: tuple[dict[str, object], ...],
) -> tuple[EvaluatorModelAttemptDiagV1, ...]:
    parsed: list[EvaluatorModelAttemptDiagV1] = []
    for event in events:
        if _event_schema_id(event) != CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA:
            continue
        payload = _event_payload(event)
        run_id = payload.get("run_id")
        node_id = payload.get("node_id")
        attempt_index = payload.get("attempt_index")
        max_iterations = payload.get("max_iterations")
        if not isinstance(run_id, str) or not isinstance(node_id, str):
            continue
        if not isinstance(attempt_index, int) or not isinstance(max_iterations, int):
            continue
        parsed.append(
            EvaluatorModelAttemptDiagV1(
                run_id=run_id,
                node_id=node_id,
                attempt_index=attempt_index,
                max_iterations=max_iterations,
            )
        )
    return tuple(parsed)


def _parse_reconciliation_events(
    events: tuple[dict[str, object], ...],
) -> tuple[ReconciliationPhaseDiagV1, ...]:
    parsed: list[ReconciliationPhaseDiagV1] = []
    for event in events:
        if _event_schema_id(event) != RECONCILIATION_PHASE_TRACE_SCHEMA:
            continue
        payload = _event_payload(event)
        run_id = payload.get("run_id")
        validation_invalid = payload.get("validation_invalid")
        entered_reconciliation = payload.get("entered_reconciliation")
        phase_raw = payload.get("phase")
        if not isinstance(run_id, str):
            continue
        if not isinstance(validation_invalid, bool) or not isinstance(
            entered_reconciliation, bool
        ):
            continue
        if not isinstance(phase_raw, str):
            continue
        try:
            phase = ReconciliationPhaseValue(phase_raw)
        except ValueError:
            continue
        parsed.append(
            ReconciliationPhaseDiagV1(
                run_id=run_id,
                validation_invalid=validation_invalid,
                entered_reconciliation=entered_reconciliation,
                phase=phase,
            )
        )
    return tuple(parsed)


def _axis_from_run(
    run_item: dict[str, object],
    *,
    passed_key: str,
    outcome_key: str,
) -> QualificationAxisStatus:
    passed = run_item.get(passed_key)
    if isinstance(passed, bool):
        return (
            QualificationAxisStatus.PASS if passed else QualificationAxisStatus.FAIL
        )
    run_result = run_item.get("run_result")
    if isinstance(run_result, dict):
        outcome = run_result.get(outcome_key)
        if outcome == "pass":
            return QualificationAxisStatus.PASS
        if outcome == "fail":
            return QualificationAxisStatus.FAIL
        if outcome == "not_evaluable":
            return QualificationAxisStatus.NOT_EVALUABLE
    return QualificationAxisStatus.UNKNOWN


def extract_run_evidence(run_item: dict[str, object]) -> QualificationRunEvidence:
    run_id = RunId(str(run_item.get("run_id", "")))
    trace_events = run_item.get("trace_events")
    events: tuple[dict[str, object], ...] = ()
    if isinstance(trace_events, list):
        events = tuple(dict(event) for event in trace_events if isinstance(event, dict))

    alignment_readback = read_typed_alignment_events(events)
    alignment_event = (
        alignment_readback.events[-1] if alignment_readback.events else None
    )
    mismatch = bool(
        alignment_event
        and alignment_event.alignment_status is AlignmentStatus.MISMATCH
    )
    direction = (
        _correction_direction_from_diag(alignment_event.alignment_direction)
        if alignment_event is not None
        else None
    )
    revision_flags = infer_alignment_revision_evidence(
        alignment_readback.events,
        _parse_attempt_events(events),
    )

    return QualificationRunEvidence(
        run_id=run_id,
        platform_status=_axis_from_run(
            run_item, passed_key="platform_passed", outcome_key="platform_outcome"
        ),
        model_status=_axis_from_run(
            run_item, passed_key="model_passed", outcome_key="model_outcome"
        ),
        evaluator_status=_axis_from_run(
            run_item,
            passed_key="evaluator_passed",
            outcome_key="evaluator_outcome",
        ),
        alignment_direction=direction,
        mismatch_detected=mismatch,
        revision_attempted=revision_flags.revision_attempted,
        typed_context_present=revision_flags.typed_context_present,
        revision_repaired=revision_flags.revision_repaired,
        attempt_events=_parse_attempt_events(events),
        reconciliation_events=_parse_reconciliation_events(events),
    )

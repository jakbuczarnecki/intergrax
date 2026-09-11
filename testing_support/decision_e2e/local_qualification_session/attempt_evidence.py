# © Artur Czarnecki. All rights reserved.

"""Canonical evaluator model attempt evidence (no heuristic third-pass counting)."""

from __future__ import annotations

from testing_support.decision_e2e.local_qualification_session.contracts import (
    CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA,
    COMPLETION_ALIGNMENT_TRACE_SCHEMA,
    AttemptEvidenceStatus,
    QualificationAttemptObservation,
    ReconciliationEvidenceStatus,
    SafetyGateOutcome,
    ThirdPassAssessment,
)
from testing_support.decision_e2e.local_qualification_session.reconciliation_leak import (
    extract_reconciliation_phase_observations,
)


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


def extract_attempt_observations(
    events: tuple[dict[str, object], ...],
) -> tuple[QualificationAttemptObservation, ...]:
    observations: list[QualificationAttemptObservation] = []
    for event in events:
        if _event_schema_id(event) != CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA:
            continue
        payload = _event_payload(event)
        run_id = payload.get("run_id")
        node_id = payload.get("node_id")
        attempt_index = payload.get("attempt_index")
        if not isinstance(run_id, str) or not isinstance(node_id, str):
            continue
        if not isinstance(attempt_index, int):
            continue
        observations.append(
            QualificationAttemptObservation(
                run_id=run_id,
                node_id=node_id,
                attempt_index=attempt_index,
            )
        )
    return tuple(observations)


def assess_third_model_pass(
    observations: tuple[QualificationAttemptObservation, ...],
    *,
    max_valid_attempt_index: int,
) -> ThirdPassAssessment:
    if not observations:
        return ThirdPassAssessment(
            outcome=SafetyGateOutcome.UNKNOWN,
            attempt_evidence_complete=False,
            violating_attempts=(),
        )
    threshold = max_valid_attempt_index + 1
    violations = tuple(
        item for item in observations if item.attempt_index >= threshold
    )
    if violations:
        return ThirdPassAssessment(
            outcome=SafetyGateOutcome.FAIL,
            attempt_evidence_complete=True,
            violating_attempts=violations,
        )
    return ThirdPassAssessment(
        outcome=SafetyGateOutcome.PASS,
        attempt_evidence_complete=True,
        violating_attempts=(),
    )


def _evaluator_model_execution_occurred(events: tuple[dict[str, object], ...]) -> bool:
    for event in events:
        schema = event.get("payload_schema_id")
        if schema == CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA:
            return True
        if schema == COMPLETION_ALIGNMENT_TRACE_SCHEMA:
            return True
    return False


def assess_run_attempt_evidence_status(
    events: tuple[dict[str, object], ...],
) -> AttemptEvidenceStatus:
    observations = extract_attempt_observations(events)
    if observations:
        return AttemptEvidenceStatus.AVAILABLE
    if not _evaluator_model_execution_occurred(events):
        return AttemptEvidenceStatus.NOT_REQUIRED
    reconciliation = extract_reconciliation_phase_observations(events)
    if reconciliation and not any(item.entered_reconciliation for item in reconciliation):
        return AttemptEvidenceStatus.NOT_REQUIRED
    if not events:
        return AttemptEvidenceStatus.NOT_REQUIRED
    return AttemptEvidenceStatus.NOT_AVAILABLE


def assess_session_attempt_evidence(
    runs: tuple[tuple[dict[str, object], ...], ...],
) -> AttemptEvidenceStatus:
    if not runs:
        return AttemptEvidenceStatus.NOT_REQUIRED
    statuses = tuple(assess_run_attempt_evidence_status(events) for events in runs)
    if any(status is AttemptEvidenceStatus.NOT_AVAILABLE for status in statuses):
        return AttemptEvidenceStatus.NOT_AVAILABLE
    if all(status is AttemptEvidenceStatus.NOT_REQUIRED for status in statuses):
        return AttemptEvidenceStatus.NOT_REQUIRED
    return AttemptEvidenceStatus.AVAILABLE


def assess_run_reconciliation_evidence_status(
    events: tuple[dict[str, object], ...],
) -> ReconciliationEvidenceStatus:
    observations = extract_reconciliation_phase_observations(events)
    if not observations:
        return ReconciliationEvidenceStatus.NOT_ENTERED
    if any(item.entered_reconciliation for item in observations):
        return ReconciliationEvidenceStatus.ENTERED
    return ReconciliationEvidenceStatus.NOT_ENTERED

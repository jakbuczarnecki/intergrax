# © Artur Czarnecki. All rights reserved.

"""Reconciliation leakage measurement from typed phase evidence."""

from __future__ import annotations

from testing_support.decision_e2e.local_qualification_session.contracts import (
    RECONCILIATION_PHASE_TRACE_SCHEMA,
    ReconciliationLeakAssessment,
    ReconciliationPhaseObservation,
    SafetyGateOutcome,
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


def extract_reconciliation_phase_observations(
    events: tuple[dict[str, object], ...],
) -> tuple[ReconciliationPhaseObservation, ...]:
    observations: list[ReconciliationPhaseObservation] = []
    for event in events:
        if _event_schema_id(event) != RECONCILIATION_PHASE_TRACE_SCHEMA:
            continue
        payload = _event_payload(event)
        run_id = payload.get("run_id")
        attempt_index = payload.get("attempt_index")
        validation_invalid = payload.get("validation_invalid")
        entered_reconciliation = payload.get("entered_reconciliation")
        if not isinstance(run_id, str) or not isinstance(attempt_index, int):
            continue
        if not isinstance(validation_invalid, bool) or not isinstance(
            entered_reconciliation, bool
        ):
            continue
        observations.append(
            ReconciliationPhaseObservation(
                run_id=run_id,
                attempt_index=attempt_index,
                validation_invalid=validation_invalid,
                entered_reconciliation=entered_reconciliation,
            )
        )
    return tuple(observations)


def assess_reconciliation_leak(
    observations: tuple[ReconciliationPhaseObservation, ...],
) -> ReconciliationLeakAssessment:
    if not observations:
        return ReconciliationLeakAssessment(
            outcome=SafetyGateOutcome.UNKNOWN,
            observations=(),
        )
    leaked = any(
        item.validation_invalid and item.entered_reconciliation for item in observations
    )
    return ReconciliationLeakAssessment(
        outcome=SafetyGateOutcome.FAIL if leaked else SafetyGateOutcome.PASS,
        observations=observations,
    )

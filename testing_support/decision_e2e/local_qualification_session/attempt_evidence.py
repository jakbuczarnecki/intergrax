# © Artur Czarnecki. All rights reserved.

"""Canonical evaluator model attempt evidence (no heuristic third-pass counting)."""

from __future__ import annotations

from testing_support.decision_e2e.local_qualification_session.contracts import (
    CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA,
    QualificationAttemptObservation,
    SafetyGateOutcome,
    ThirdPassAssessment,
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

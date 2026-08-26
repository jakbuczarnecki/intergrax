# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic lifecycle invariant analysis on execution reconstruction (DIAG-3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.execution_identity import AttemptId, EventId, RunId, TaskId
from intergrax.runtime.diagnostics.execution_reconstruction import (
    ExecutionReconstruction,
    RuntimeHistoryCompleteness,
    ReconstructedAttempt,
)
from intergrax.runtime.events.asof_projection import (
    apply_lifecycle_event,
    InvalidRunExecutionHistoryError,
    is_final_run_lifecycle_status,
    RunExecutionLifecycleStatus,
    RunLifecycleViolationKind,
)
from intergrax.runtime.events.execution_position import ExecutionEventPosition, PositionedRuntimeEvent
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.observability.causal_evidence import PlatformCausalEvidence
from intergrax.runtime.observability.causal_evidence_persistence import causal_evidence_query_order_key


class LifecycleAnalysisIntegrityError(Exception):
    """Raised when lifecycle analysis encounters inconsistent typed facts."""


class LifecycleAnomalyKind(StrEnum):
    """Factual lifecycle inconsistency kinds supported by DIAG-3 v1."""

    CAUSAL_ATTEMPT_WITHOUT_RUNTIME_HISTORY = "causal_attempt_without_runtime_history"
    RUNTIME_ATTEMPT_WITHOUT_CAUSAL_EVIDENCE = "runtime_attempt_without_causal_evidence"
    RUNTIME_HISTORY_TRUNCATED = "runtime_history_truncated"
    MULTIPLE_TERMINAL_OUTCOMES = "multiple_terminal_outcomes"
    EVENT_AFTER_TERMINAL = "event_after_terminal"
    DISALLOWED_AFTER_FAILED = "disallowed_after_failed"


class LifecycleAnomalyScope(StrEnum):
    """Whether a finding applies to the whole execution or one attempt."""

    EXECUTION = "execution"
    ATTEMPT = "attempt"


_LIFECYCLE_VIOLATION_TO_ANOMALY_KIND: dict[
    RunLifecycleViolationKind, LifecycleAnomalyKind
] = {
    RunLifecycleViolationKind.CONFLICTING_FINAL_OUTCOME: (
        LifecycleAnomalyKind.MULTIPLE_TERMINAL_OUTCOMES
    ),
    RunLifecycleViolationKind.EVENT_AFTER_TERMINAL: LifecycleAnomalyKind.EVENT_AFTER_TERMINAL,
    RunLifecycleViolationKind.DISALLOWED_AFTER_FAILED: (
        LifecycleAnomalyKind.DISALLOWED_AFTER_FAILED
    ),
}

LIFECYCLE_TRANSITION_ANOMALY_KINDS: frozenset[LifecycleAnomalyKind] = frozenset(
    {
        LifecycleAnomalyKind.MULTIPLE_TERMINAL_OUTCOMES,
        LifecycleAnomalyKind.EVENT_AFTER_TERMINAL,
        LifecycleAnomalyKind.DISALLOWED_AFTER_FAILED,
    }
)


@dataclass(frozen=True, slots=True)
class LifecycleViolationTransition:
    """Typed TRACE-ASOF lifecycle violation descriptor — not a lifecycle model."""

    violation_kind: RunLifecycleViolationKind
    prior_status: RunExecutionLifecycleStatus
    violating_event_type: RuntimeEventType


@dataclass(frozen=True, slots=True)
class LifecycleAnomaly:
    """One deterministic invariant violation derived from reconstruction facts."""

    kind: LifecycleAnomalyKind
    scope: LifecycleAnomalyScope
    attempt_id: AttemptId | None
    supporting_event_ids: tuple[EventId, ...]
    supporting_evidence_ids: tuple[EventId, ...]
    supporting_positions: tuple[ExecutionEventPosition, ...]
    factual_message: str
    lifecycle_transition: LifecycleViolationTransition | None = None


@dataclass(frozen=True, slots=True)
class LifecycleAnalysis:
    """Derived lifecycle invariant findings for one execution reconstruction."""

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    anomalies: tuple[LifecycleAnomaly, ...]

    @property
    def has_anomalies(self) -> bool:
        return bool(self.anomalies)


class LifecycleAnomalyAnalyzer:
    """
    Deterministic invariant checker over ``ExecutionReconstruction``.

    No persistence, no diagnosis, no root-cause inference.
    """

    def analyze(self, reconstruction: ExecutionReconstruction) -> LifecycleAnalysis:
        anomalies: list[LifecycleAnomaly] = []

        if (
            reconstruction.runtime_history_completeness
            is RuntimeHistoryCompleteness.TRUNCATED
        ):
            anomalies.append(_runtime_history_truncated_anomaly(reconstruction))

        for attempt in reconstruction.attempts:
            anomalies.extend(_attempt_evidence_anomalies(reconstruction, attempt))

        positioned = _sorted_positioned_events(reconstruction.positioned_events)
        anomalies.extend(_lifecycle_violation_anomalies(positioned))

        return LifecycleAnalysis(
            tenant_id=reconstruction.tenant_id,
            task_id=reconstruction.task_id,
            run_id=reconstruction.run_id,
            anomalies=_sort_anomalies(anomalies, reconstruction),
        )


def _runtime_history_truncated_anomaly(
    reconstruction: ExecutionReconstruction,
) -> LifecycleAnomaly:
    return LifecycleAnomaly(
        kind=LifecycleAnomalyKind.RUNTIME_HISTORY_TRUNCATED,
        scope=LifecycleAnomalyScope.EXECUTION,
        attempt_id=None,
        supporting_event_ids=(),
        supporting_evidence_ids=(),
        supporting_positions=(),
        factual_message=(
            "Runtime event history for this execution was truncated during "
            "reconstruction; full lifecycle completeness cannot be established."
        ),
    )


def _attempt_evidence_anomalies(
    reconstruction: ExecutionReconstruction,
    attempt: ReconstructedAttempt,
) -> tuple[LifecycleAnomaly, ...]:
    findings: list[LifecycleAnomaly] = []

    if attempt.has_transport_evidence and not attempt.has_runtime_events:
        evidence_ids = tuple(evidence.evidence_id for evidence in attempt.causal_evidence)
        findings.append(
            LifecycleAnomaly(
                kind=LifecycleAnomalyKind.CAUSAL_ATTEMPT_WITHOUT_RUNTIME_HISTORY,
                scope=LifecycleAnomalyScope.ATTEMPT,
                attempt_id=attempt.attempt_id,
                supporting_event_ids=(),
                supporting_evidence_ids=evidence_ids,
                supporting_positions=(),
                factual_message=(
                    f"Attempt {attempt.attempt_id} has transport→execution causal "
                    "evidence but no reconstructed RuntimeEvent history."
                ),
            )
        )

    if (
        reconstruction.has_transport_evidence
        and attempt.has_runtime_events
        and not attempt.has_transport_evidence
    ):
        event_ids = tuple(row.event.event_id for row in attempt.positioned_events)
        positions = tuple(row.position for row in attempt.positioned_events)
        findings.append(
            LifecycleAnomaly(
                kind=LifecycleAnomalyKind.RUNTIME_ATTEMPT_WITHOUT_CAUSAL_EVIDENCE,
                scope=LifecycleAnomalyScope.ATTEMPT,
                attempt_id=attempt.attempt_id,
                supporting_event_ids=event_ids,
                supporting_evidence_ids=(),
                supporting_positions=positions,
                factual_message=(
                    f"Attempt {attempt.attempt_id} has RuntimeEvent history but no "
                    "transport→execution causal evidence while other attempts in the "
                    "same execution do."
                ),
            )
        )

    return tuple(findings)


def _is_terminal_publish_marker(event: RuntimeEvent) -> bool:
    """Nexus terminal bookkeeping replay — not an operational lifecycle violation."""
    payload = event.payload or {}
    return (
        payload.get("source") == "task_lifecycle"
        and payload.get("message") == "task terminal"
    )


def _lifecycle_violation_anomalies(
    positioned_events: tuple[PositionedRuntimeEvent, ...],
) -> tuple[LifecycleAnomaly, ...]:
    if not positioned_events:
        return ()

    findings: list[LifecycleAnomaly] = []
    status = RunExecutionLifecycleStatus.CREATED
    last_status_event: PositionedRuntimeEvent | None = None
    last_terminal_event: PositionedRuntimeEvent | None = None

    for row in positioned_events:
        if _is_terminal_publish_marker(row.event):
            continue
        try:
            new_status = apply_lifecycle_event(status, row.event.event_type)
        except InvalidRunExecutionHistoryError as exc:
            if exc.kind is None:
                raise
            prior_event = _prior_lifecycle_event_for_violation(
                exc.kind,
                last_terminal_event=last_terminal_event,
                last_status_event=last_status_event,
            )
            scope = _lifecycle_violation_scope(
                row,
                prior_event=prior_event,
            )
            supporting_event_ids, supporting_positions = _lifecycle_violation_provenance(
                prior_event,
                row,
            )
            lifecycle_transition = _lifecycle_violation_descriptor_from_exc(exc)
            findings.append(
                LifecycleAnomaly(
                    kind=_LIFECYCLE_VIOLATION_TO_ANOMALY_KIND[exc.kind],
                    scope=scope,
                    attempt_id=row.event.attempt_id if scope is LifecycleAnomalyScope.ATTEMPT else None,
                    supporting_event_ids=supporting_event_ids,
                    supporting_positions=supporting_positions,
                    supporting_evidence_ids=(),
                    factual_message=str(exc),
                    lifecycle_transition=lifecycle_transition,
                )
            )
            continue

        if new_status != status:
            last_status_event = row
        status = new_status
        if is_final_run_lifecycle_status(status):
            last_terminal_event = row

    return tuple(findings)


def _lifecycle_violation_descriptor_from_exc(
    exc: InvalidRunExecutionHistoryError,
) -> LifecycleViolationTransition:
    if exc.kind is None:
        raise LifecycleAnalysisIntegrityError(
            "typed lifecycle violation missing violation kind"
        )
    if exc.current_status is None or exc.event_type is None:
        raise LifecycleAnalysisIntegrityError(
            "typed lifecycle violation missing current_status or event_type"
        )
    return LifecycleViolationTransition(
        violation_kind=exc.kind,
        prior_status=exc.current_status,
        violating_event_type=exc.event_type,
    )


def _prior_lifecycle_event_for_violation(
    kind: RunLifecycleViolationKind,
    *,
    last_terminal_event: PositionedRuntimeEvent | None,
    last_status_event: PositionedRuntimeEvent | None,
) -> PositionedRuntimeEvent | None:
    if kind is RunLifecycleViolationKind.EVENT_AFTER_TERMINAL:
        return last_terminal_event
    return last_status_event


def _lifecycle_violation_provenance(
    prior_event: PositionedRuntimeEvent | None,
    violating_event: PositionedRuntimeEvent,
) -> tuple[tuple[EventId, ...], tuple[ExecutionEventPosition, ...]]:
    if prior_event is None:
        return (violating_event.event.event_id,), (violating_event.position,)
    return (
        (prior_event.event.event_id, violating_event.event.event_id),
        (prior_event.position, violating_event.position),
    )


def _lifecycle_violation_scope(
    violating_event: PositionedRuntimeEvent,
    *,
    prior_event: PositionedRuntimeEvent | None,
) -> LifecycleAnomalyScope:
    if prior_event is None:
        return LifecycleAnomalyScope.ATTEMPT
    if (
        violating_event.event.attempt_id
        == prior_event.event.attempt_id
    ):
        return LifecycleAnomalyScope.ATTEMPT
    return LifecycleAnomalyScope.EXECUTION


def _sorted_positioned_events(
    positioned_events: tuple[PositionedRuntimeEvent, ...],
) -> tuple[PositionedRuntimeEvent, ...]:
    return tuple(sorted(positioned_events, key=lambda row: row.position.value))


def _sort_anomalies(
    anomalies: list[LifecycleAnomaly],
    reconstruction: ExecutionReconstruction,
) -> tuple[LifecycleAnomaly, ...]:
    evidence_by_id = {
        evidence.evidence_id: evidence for evidence in reconstruction.causal_evidence
    }
    return tuple(
        sorted(
            anomalies,
            key=lambda anomaly: _anomaly_presentation_order_key(anomaly, evidence_by_id),
        )
    )


def _anomaly_presentation_order_key(
    anomaly: LifecycleAnomaly,
    evidence_by_id: dict[EventId, PlatformCausalEvidence],
) -> tuple:
    if anomaly.supporting_positions:
        position_key = (0, min(position.value for position in anomaly.supporting_positions))
    else:
        position_key = (1, 0)

    evidence_rows = tuple(
        evidence_by_id[evidence_id]
        for evidence_id in anomaly.supporting_evidence_ids
        if evidence_id in evidence_by_id
    )
    if evidence_rows:
        recorded_at, evidence_id = causal_evidence_query_order_key(
            min(evidence_rows, key=causal_evidence_query_order_key)
        )
        evidence_key = (0, recorded_at, evidence_id)
    else:
        evidence_key = (1, 0, "")

    attempt_key = str(anomaly.attempt_id) if anomaly.attempt_id is not None else ""
    return (position_key, evidence_key, anomaly.kind.value, attempt_key)

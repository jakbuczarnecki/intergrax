# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic execution failure analysis over canonical RuntimeEvent evidence (DIAG R2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.runtime.diagnostics.diagnostic_precision import (
    DiagnosticCertainty,
    DiagnosticPrecision,
    FailureBoundary,
)
from intergrax.runtime.diagnostics.execution_lineage_reconstruction import (
    ExecutionLineageCompleteness,
    ExecutionLineageReadStatus,
    ReconstructedAttemptLineage,
)
from intergrax.runtime.diagnostics.execution_reconstruction import (
    ExecutionReconstruction,
    ReconstructedAttempt,
)
from intergrax.runtime.events.payload_registry import (
    RuntimeEventPayloadError,
    validate_payload_envelope,
)
from intergrax.runtime.events.payloads.canonical import ExecutionFailurePayloadV1
from intergrax.runtime.events.runtime_event import RuntimeEventType

if TYPE_CHECKING:
    from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticFinding


class ExecutionFailureAnalysisIntegrityError(Exception):
    """Raised when EXECUTION_FAILED evidence is present but not structurally valid."""


_EXECUTION_FAILED_CLAIM = (
    "Canonical runtime evidence proves that this execution failed."
)


def _lineage_execution_ids(lineage: ReconstructedAttemptLineage) -> frozenset[str]:
    ids: set[str] = set()
    for segment in lineage.segments:
        for admission in segment.admissions:
            ids.add(admission.execution_id)
    return frozenset(ids)


def _validate_lineage_membership(
    execution_id: str,
    attempt: ReconstructedAttempt,
) -> None:
    lineage = attempt.lineage
    if lineage is None:
        return
    if lineage.read_status is not ExecutionLineageReadStatus.AVAILABLE:
        return
    if lineage.completeness is not ExecutionLineageCompleteness.COMPLETE:
        return
    if execution_id not in _lineage_execution_ids(lineage):
        raise ExecutionFailureAnalysisIntegrityError(
            "EXECUTION_FAILED execution_id missing from complete attempt lineage",
        )


class ExecutionFailureAnalyzer:
    """Derives proven execution-level failure findings from reconstruction only."""

    def analyze(
        self,
        reconstruction: ExecutionReconstruction,
    ) -> tuple[DiagnosticFinding, ...]:
        from intergrax.runtime.diagnostics.diagnostic_assessment import (
            DiagnosticFinding,
            DiagnosticFindingKind,
        )
        from intergrax.runtime.diagnostics.lifecycle_analysis import (
            LifecycleAnomalyScope,
        )

        findings: list[DiagnosticFinding] = []
        attempts_by_id = {
            attempt.attempt_id: attempt for attempt in reconstruction.attempts
        }

        for positioned in reconstruction.positioned_events:
            event = positioned.event
            if event.event_type is not RuntimeEventType.EXECUTION_FAILED:
                continue
            attempt = attempts_by_id.get(event.attempt_id)
            payload = _require_execution_failure_payload(event.payload)
            if attempt is not None:
                _validate_lineage_membership(event.execution_id, attempt)

            boundary = FailureBoundary(
                execution_id=event.execution_id,
                supporting_event_id=event.event_id,
                certainty=DiagnosticCertainty.PROVEN,
                precision=DiagnosticPrecision.EXECUTION_LEVEL,
            )
            findings.append(
                DiagnosticFinding(
                    kind=DiagnosticFindingKind.EXECUTION_FAILED,
                    scope=LifecycleAnomalyScope.ATTEMPT,
                    attempt_id=event.attempt_id,
                    certainty=DiagnosticCertainty.PROVEN,
                    claim=_EXECUTION_FAILED_CLAIM,
                    source_anomaly_kind=None,
                    supporting_event_ids=(event.event_id,),
                    supporting_evidence_ids=(),
                    supporting_positions=(positioned.position,),
                    lifecycle_transition=None,
                    execution_id=event.execution_id,
                    precision=DiagnosticPrecision.EXECUTION_LEVEL,
                    failure_boundary=boundary,
                    execution_failure_kind=payload.failure_kind,
                ),
            )
        return tuple(findings)


def _require_execution_failure_payload(
    payload: dict[str, object],
) -> ExecutionFailurePayloadV1:
    try:
        parsed = validate_payload_envelope(payload)
    except RuntimeEventPayloadError as exc:
        raise ExecutionFailureAnalysisIntegrityError(
            "malformed EXECUTION_FAILED typed payload",
        ) from exc
    if parsed is None:
        raise ExecutionFailureAnalysisIntegrityError(
            "EXECUTION_FAILED missing typed payload envelope",
        )
    if not isinstance(parsed, ExecutionFailurePayloadV1):
        raise ExecutionFailureAnalysisIntegrityError(
            "EXECUTION_FAILED payload schema is not execution_failure.v1",
        )
    return parsed


__all__ = [
    "ExecutionFailureAnalysisIntegrityError",
    "ExecutionFailureAnalyzer",
]

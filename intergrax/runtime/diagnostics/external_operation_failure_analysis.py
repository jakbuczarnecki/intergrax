# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic external operation failure analysis (LLM external op DIAG R2)."""

from __future__ import annotations

from intergrax.runtime.diagnostics.diagnostic_precision import (
    DiagnosticCertainty,
    DiagnosticPrecision,
    FailureBoundary,
)
from intergrax.runtime.events.payload_registry import (
    RuntimeEventPayloadError,
    validate_payload_envelope,
)
from intergrax.runtime.events.payloads.canonical import ExternalOperationFailurePayloadV1
from intergrax.runtime.events.runtime_event import RuntimeEventType


class ExternalOperationFailureAnalysisIntegrityError(Exception):
    """Raised when EXTERNAL_OPERATION_FAILED evidence is invalid."""


_EXTERNAL_OPERATION_FAILED_CLAIM = (
    "Canonical runtime evidence proves that an external operation failed at this execution."
)


class ExternalOperationFailureAnalyzer:
    def analyze(self, reconstruction) -> tuple:  # noqa: ANN001
        from intergrax.runtime.diagnostics.diagnostic_assessment import (
            DiagnosticFinding,
            DiagnosticFindingKind,
        )
        from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyScope

        findings: list[DiagnosticFinding] = []
        for positioned in reconstruction.positioned_events:
            event = positioned.event
            if event.event_type is not RuntimeEventType.EXTERNAL_OPERATION_FAILED:
                continue
            payload = _require_external_operation_failure_payload(event.payload)
            boundary = FailureBoundary(
                execution_id=event.execution_id,
                supporting_event_id=event.event_id,
                certainty=DiagnosticCertainty.PROVEN,
                precision=DiagnosticPrecision.EXTERNAL_BOUNDARY,
            )
            findings.append(
                DiagnosticFinding(
                    kind=DiagnosticFindingKind.EXTERNAL_OPERATION_FAILED,
                    scope=LifecycleAnomalyScope.ATTEMPT,
                    attempt_id=event.attempt_id,
                    certainty=DiagnosticCertainty.PROVEN,
                    claim=_EXTERNAL_OPERATION_FAILED_CLAIM,
                    source_anomaly_kind=None,
                    supporting_event_ids=(event.event_id,),
                    supporting_evidence_ids=(),
                    supporting_positions=(positioned.position,),
                    execution_id=event.execution_id,
                    precision=DiagnosticPrecision.EXTERNAL_BOUNDARY,
                    failure_boundary=boundary,
                    external_operation_failure_kind=payload.failure_kind,
                    operation_attempt_id=payload.operation_attempt_id,
                    provider_id=payload.provider_id,
                ),
            )
        return tuple(findings)


def _require_external_operation_failure_payload(
    payload: dict[str, object],
) -> ExternalOperationFailurePayloadV1:
    try:
        parsed = validate_payload_envelope(payload)
    except RuntimeEventPayloadError as exc:
        raise ExternalOperationFailureAnalysisIntegrityError(
            "malformed EXTERNAL_OPERATION_FAILED typed payload",
        ) from exc
    if parsed is None:
        raise ExternalOperationFailureAnalysisIntegrityError(
            "EXTERNAL_OPERATION_FAILED missing typed payload envelope",
        )
    if not isinstance(parsed, ExternalOperationFailurePayloadV1):
        raise ExternalOperationFailureAnalysisIntegrityError(
            "EXTERNAL_OPERATION_FAILED payload schema is not external_operation_failure.v1",
        )
    return parsed


__all__ = [
    "ExternalOperationFailureAnalysisIntegrityError",
    "ExternalOperationFailureAnalyzer",
]

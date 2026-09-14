# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Map contract export payloads to runtime export envelopes (OTLP adapter seam)."""

from __future__ import annotations

from datetime import datetime, timezone

from intergrax.contracts.event_delivery import (
    ObservabilityExportPayload,
    ObservabilityExportSafeAttribute,
)
from intergrax.runtime.observability.export_boundary import (
    ExportRecordKind,
    ExportStatus,
    ObservabilityExportEnvelope,
)


def _safe_attributes_to_counts(
    safe_attributes: tuple[ObservabilityExportSafeAttribute, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in safe_attributes:
        if item.key in ("hit_count", "event_count", "parser_trace_count") and isinstance(
            item.value,
            int,
        ):
            counts[item.key] = item.value
    return counts


def _status_from_safe_attributes(
    safe_attributes: tuple[ObservabilityExportSafeAttribute, ...],
) -> ExportStatus:
    for item in safe_attributes:
        if item.key == "status" and isinstance(item.value, str):
            try:
                return ExportStatus(item.value)
            except ValueError:
                return ExportStatus.UNKNOWN
    return ExportStatus.UNKNOWN


def envelope_from_observability_export_payload(
    payload: ObservabilityExportPayload,
) -> ObservabilityExportEnvelope:
    safe_attributes = payload.safe_attributes
    tool_id = ""
    capability = ""
    latency_ms: int | None = None
    sha256 = ""
    for item in safe_attributes:
        if item.key == "tool_id" and isinstance(item.value, str):
            tool_id = item.value
        elif item.key == "capability" and isinstance(item.value, str):
            capability = item.value
        elif item.key in ("latency_ms", "duration_ms") and isinstance(item.value, int):
            latency_ms = item.value
        elif item.key == "args_digest" and isinstance(item.value, str):
            sha256 = item.value

    return ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        recorded_at=datetime.now(timezone.utc),
        run_id=payload.run_id,
        task_id=payload.task_id,
        attempt_id=payload.attempt_id,
        execution_id=payload.execution_id,
        agent_id=payload.agent_id,
        capability=capability,
        tool_id=tool_id,
        event_type=payload.event_type,
        status=_status_from_safe_attributes(safe_attributes),
        latency_ms=latency_ms,
        counts=_safe_attributes_to_counts(safe_attributes),
        sha256=sha256,
        tenant_id=payload.tenant_id,
        correlation_id=payload.correlation_id,
        event_id=payload.event_id,
        parent_event_id=payload.parent_event_id,
        execution_phase=payload.execution_phase,
        w3c_traceparent=payload.w3c_traceparent,
        w3c_tracestate=payload.w3c_tracestate,
        source_schema_id="runtime_event_export_source.v1",
    )

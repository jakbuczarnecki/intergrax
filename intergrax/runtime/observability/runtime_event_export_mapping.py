# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pure RuntimeEvent → RuntimeEventExportSource mapping (import-cycle seam)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.runtime.observability.runtime_event_export_models import (
    RuntimeEventExportSource,
)

if TYPE_CHECKING:
    from intergrax.runtime.events.runtime_event import RuntimeEvent

_SAFE_RUNTIME_EVENT_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "tool_id",
        "capability",
        "latency_ms",
        "duration_ms",
        "hit_count",
        "error_code",
        "policy_rule_id",
        "args_digest",
        "collection_id",
        "payload_schema_id",
        "schema_id",
        "event_count",
        "parser_trace_count",
        "status",
    }
)


def _extract_safe_payload(payload: object) -> dict[str, str | int]:
    if not isinstance(payload, dict):
        return {}
    safe: dict[str, str | int] = {}
    for key, value in payload.items():
        if key not in _SAFE_RUNTIME_EVENT_PAYLOAD_KEYS:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            safe[key] = value
        elif isinstance(value, str):
            safe[key] = value
    return safe


def runtime_event_export_source_from_event(
    event: RuntimeEvent,
) -> RuntimeEventExportSource:
    safe_payload = _extract_safe_payload(event.payload)
    return RuntimeEventExportSource(
        event_id=event.event_id,
        run_id=event.run_id,
        task_id=event.task_id,
        attempt_id=str(event.attempt_id),
        execution_id=str(event.execution_id),
        event_type=event.event_type.value,
        agent_id=event.agent_id or "",
        tenant_id=event.tenant_id or "",
        correlation_id=event.correlation_id,
        occurred_at=event.timestamp,
        execution_phase=event.phase.value,
        parent_event_id=str(event.parent_event_id or ""),
        w3c_traceparent=event.traceparent or "",
        w3c_tracestate=event.tracestate or "",
        safe_payload=safe_payload,
    )

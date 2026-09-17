# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Project ToolRuntime spine events into read-model invocation records."""

from __future__ import annotations

from intergrax.contracts.execution_reconstruction import ExecutionReconstruction
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.contracts.tool_runtime_read import (
    ToolRuntimeInvocationOutcome,
    ToolRuntimeInvocationReadResult,
    ToolRuntimeInvocationRecord,
)
from intergrax.runtime.runtime_inspection.redaction import sanitize_inspection_text

_TOOL_OUTCOME_TYPES: frozenset[RuntimeEventType] = frozenset(
    {
        RuntimeEventType.TOOL_COMPLETED,
        RuntimeEventType.TOOL_DENIED,
        RuntimeEventType.TOOL_FAILED,
    },
)


def project_tool_invocations_from_reconstruction(
    reconstruction: ExecutionReconstruction,
    *,
    limit: int,
) -> ToolRuntimeInvocationReadResult:
    if limit < 1:
        raise ValueError("limit must be positive")

    open_requests: list[tuple[int, str, str]] = []
    records: list[ToolRuntimeInvocationRecord] = []

    for positioned in reconstruction.positioned_events:
        event = positioned.event
        if event.event_type is RuntimeEventType.TOOL_REQUESTED:
            tool_id = str(event.payload.get("tool_id") or event.event_kind or "unknown_tool")
            invocation_id = str(event.event_id)
            open_requests.append((positioned.position.value, invocation_id, tool_id))
            continue
        if event.event_type not in _TOOL_OUTCOME_TYPES:
            continue
        if not open_requests:
            continue
        sequence_key, invocation_id, tool_id = open_requests.pop(0)
        payload = event.payload
        tool_id = str(payload.get("tool_id") or tool_id)
        status_label = str(payload.get("status") or event.event_type.value)
        outcome = _outcome_for_event_type(event.event_type)
        failure_classification = None
        if outcome is ToolRuntimeInvocationOutcome.FAILED:
            failure_classification = str(payload.get("error_code") or "tool_failed")
        if outcome is ToolRuntimeInvocationOutcome.DENIED:
            failure_classification = str(payload.get("error_code") or "tool_denied")
        args_digest = payload.get("args_digest")
        args_digest_ref = str(args_digest) if args_digest else None
        correlation = event.correlation_id.strip() if event.correlation_id else None
        safe_summary = sanitize_inspection_text(
            f"{tool_id}:{status_label}",
        )
        records.append(
            ToolRuntimeInvocationRecord(
                invocation_id=invocation_id,
                tool_id=tool_id,
                execution_id=event.execution_id,
                attempt_id=event.attempt_id,
                sequence_key=sequence_key,
                outcome=outcome,
                status_label=status_label,
                failure_classification=failure_classification,
                args_digest_ref=args_digest_ref,
                provider_correlation_ref=correlation,
                governance_evidence_refs=(),
                evidence_refs=(str(event.event_id),),
                safe_summary=safe_summary,
            ),
        )
        if len(records) >= limit:
            break

    truncated = len(records) >= limit
    records.sort(key=lambda item: (item.sequence_key, item.invocation_id))
    return ToolRuntimeInvocationReadResult(records=tuple(records), is_truncated=truncated)


def _outcome_for_event_type(event_type: RuntimeEventType) -> ToolRuntimeInvocationOutcome:
    if event_type is RuntimeEventType.TOOL_COMPLETED:
        return ToolRuntimeInvocationOutcome.COMPLETED
    if event_type is RuntimeEventType.TOOL_DENIED:
        return ToolRuntimeInvocationOutcome.DENIED
    return ToolRuntimeInvocationOutcome.FAILED


__all__ = ["project_tool_invocations_from_reconstruction"]

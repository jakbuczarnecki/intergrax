# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Aggregate external operation failures for predictive intelligence (R2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import TaskId
from intergrax.contracts.external_operations.failure import ExternalOperationFailureKind
from intergrax.runtime.events.payload_registry import validate_payload_envelope
from intergrax.runtime.events.payloads.canonical import ExternalOperationFailurePayloadV1
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEventType


@dataclass(frozen=True, slots=True)
class ExternalOperationFailureHistoryEntry:
    provider_id: str
    failure_kind: ExternalOperationFailureKind
    operation_type: str
    runtime_event_id: str


def collect_external_operation_failure_history(
    store: RuntimeEventPersistence,
    *,
    tenant_id: str,
    task_id: TaskId,
    provider_id: str | None = None,
    limit: int = 500,
) -> tuple[ExternalOperationFailureHistoryEntry, ...]:
    """Read-only history slice for predictive risk analyzers — no provider control."""
    entries: list[ExternalOperationFailureHistoryEntry] = []
    for event in store.list_for_task(str(task_id), tenant_id=tenant_id, limit=limit):
        if event.event_type is not RuntimeEventType.EXTERNAL_OPERATION_FAILED:
            continue
        parsed = validate_payload_envelope(event.payload)
        if not isinstance(parsed, ExternalOperationFailurePayloadV1):
            continue
        if provider_id is not None and parsed.provider_id != provider_id:
            continue
        entries.append(
            ExternalOperationFailureHistoryEntry(
                provider_id=parsed.provider_id,
                failure_kind=parsed.failure_kind,
                operation_type=parsed.operation_type,
                runtime_event_id=str(event.event_id),
            ),
        )
    return tuple(entries)


__all__ = [
    "ExternalOperationFailureHistoryEntry",
    "collect_external_operation_failure_history",
]

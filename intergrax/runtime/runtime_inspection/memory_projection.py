# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Project memory spine events into read-model operation records."""

from __future__ import annotations

from intergrax.contracts.execution_reconstruction import ExecutionReconstruction
from intergrax.contracts.memory_runtime_read import (
    MemoryRuntimeOperationClass,
    MemoryRuntimeOperationReadResult,
    MemoryRuntimeOperationRecord,
    MemoryRuntimeOperationStatus,
)
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.runtime.runtime_inspection.redaction import sanitize_inspection_text


def project_memory_operations_from_reconstruction(
    reconstruction: ExecutionReconstruction,
    *,
    limit: int,
) -> MemoryRuntimeOperationReadResult:
    if limit < 1:
        raise ValueError("limit must be positive")

    records: list[MemoryRuntimeOperationRecord] = []
    for positioned in reconstruction.positioned_events:
        event = positioned.event
        if event.event_type is RuntimeEventType.MEMORY_READ:
            operation_class = MemoryRuntimeOperationClass.READ
        elif event.event_type is RuntimeEventType.MEMORY_WRITE:
            operation_class = MemoryRuntimeOperationClass.WRITE
        else:
            continue
        payload = event.payload
        namespace = str(payload.get("namespace") or "default").strip() or "default"
        key = str(payload.get("key") or "unknown").strip() or "unknown"
        found = bool(payload.get("found"))
        if operation_class is MemoryRuntimeOperationClass.READ:
            status = (
                MemoryRuntimeOperationStatus.HIT
                if found
                else MemoryRuntimeOperationStatus.MISS
            )
        else:
            status = MemoryRuntimeOperationStatus.RECORDED
        record_id = payload.get("record_id")
        record_ref = str(record_id).strip() if record_id else None
        safe_summary = sanitize_inspection_text(f"{operation_class.value}:{namespace}/{key}")
        records.append(
            MemoryRuntimeOperationRecord(
                operation_ref=str(event.event_id),
                memory_class=namespace,
                operation_class=operation_class,
                operation_status=status,
                record_ref=record_ref,
                source_category="task_memory_spine",
                tenant_id=event.tenant_id or reconstruction.tenant_id,
                task_id=event.task_id,
                run_id=event.run_id,
                execution_id=event.execution_id,
                attempt_id=event.attempt_id,
                sequence_key=positioned.position.value,
                evidence_refs=(str(event.event_id),),
                safe_summary=safe_summary,
            ),
        )
        if len(records) >= limit:
            break

    sorted_records = tuple(
        sorted(records, key=lambda item: (item.sequence_key, item.operation_ref)),
    )
    total_matching = sum(
        1
        for positioned in reconstruction.positioned_events
        if positioned.event.event_type
        in (RuntimeEventType.MEMORY_READ, RuntimeEventType.MEMORY_WRITE)
    )
    return MemoryRuntimeOperationReadResult(
        records=sorted_records,
        is_truncated=total_matching > limit,
    )


__all__ = ["project_memory_operations_from_reconstruction"]

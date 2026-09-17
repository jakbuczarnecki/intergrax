# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Project external operation spine events into external work read-model facts."""

from __future__ import annotations

from intergrax.contracts.execution_reconstruction import ExecutionReconstruction
from intergrax.contracts.external_work_runtime_read import (
    ExternalWorkRuntimeFactReadResult,
    ExternalWorkRuntimeFactRecord,
    ExternalWorkRuntimeStatus,
)
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.runtime.events.payload_registry import (
    RuntimeEventPayloadError,
    validate_payload_envelope,
)
from intergrax.runtime.events.payloads.canonical import ExternalOperationFailurePayloadV1
from intergrax.runtime.runtime_inspection.redaction import sanitize_inspection_text


def project_external_work_from_reconstruction(
    reconstruction: ExecutionReconstruction,
    *,
    limit: int,
) -> ExternalWorkRuntimeFactReadResult:
    if limit < 1:
        raise ValueError("limit must be positive")

    records: list[ExternalWorkRuntimeFactRecord] = []
    total_matching = 0
    for positioned in reconstruction.positioned_events:
        event = positioned.event
        if event.event_type is not RuntimeEventType.EXTERNAL_OPERATION_FAILED:
            continue
        total_matching += 1
        if len(records) >= limit:
            continue
        payload = _require_external_operation_failure_payload(event.payload)
        failure_kind = payload.failure_kind.value
        safe_summary = sanitize_inspection_text(
            f"{payload.provider_id}:{payload.operation_type}:{failure_kind}",
        )
        records.append(
            ExternalWorkRuntimeFactRecord(
                work_ref=payload.operation_attempt_id,
                work_class=payload.operation_type,
                work_status=ExternalWorkRuntimeStatus.FAILED,
                provider_ref=payload.provider_id,
                failure_classification=failure_kind,
                retryable=payload.retryable,
                tenant_id=event.tenant_id or reconstruction.tenant_id,
                task_id=event.task_id,
                run_id=event.run_id,
                execution_id=event.execution_id,
                attempt_id=event.attempt_id,
                sequence_key=positioned.position.value,
                evidence_refs=tuple(payload.evidence_refs) or (str(event.event_id),),
                safe_summary=safe_summary,
            ),
        )

    sorted_records = tuple(
        sorted(records, key=lambda item: (item.sequence_key, item.work_ref)),
    )
    return ExternalWorkRuntimeFactReadResult(
        records=sorted_records,
        is_truncated=total_matching > limit,
    )


def _require_external_operation_failure_payload(
    payload: dict[str, object],
) -> ExternalOperationFailurePayloadV1:
    try:
        parsed = validate_payload_envelope(payload)
    except RuntimeEventPayloadError as exc:
        raise ValueError("malformed EXTERNAL_OPERATION_FAILED typed payload") from exc
    if parsed is None:
        raise ValueError("EXTERNAL_OPERATION_FAILED missing typed payload envelope")
    if not isinstance(parsed, ExternalOperationFailurePayloadV1):
        raise ValueError("EXTERNAL_OPERATION_FAILED payload schema mismatch")
    return parsed


__all__ = ["project_external_work_from_reconstruction"]

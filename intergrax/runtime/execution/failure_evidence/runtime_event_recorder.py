# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""RuntimeEvent bus adapter for execution failure evidence (DIAG R2)."""

from __future__ import annotations

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_failure_evidence import (
    ExecutionFailureEvidenceRecordResult,
    ExecutionFailureEvidenceRecordStatus,
    ExecutionFailureEvidenceRequest,
    validate_execution_failure_evidence_request,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_catalog import get_catalog_entry
from intergrax.runtime.events.payload_registry import runtime_event_with_payload
from intergrax.runtime.events.payloads.canonical import ExecutionFailurePayloadV1
from intergrax.runtime.events.persistence_contract import (
    MandatoryEvidencePersistenceError,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType


class RuntimeEventExecutionFailureEvidenceRecorder:
    """Maps execution failure evidence requests onto the canonical RuntimeEvent bus."""

    __slots__ = ("_bus",)

    def __init__(self, bus: RuntimeEventBus) -> None:
        self._bus = bus

    def record_failure(
        self,
        request: ExecutionFailureEvidenceRequest,
    ) -> ExecutionFailureEvidenceRecordResult:
        validated = validate_execution_failure_evidence_request(request)
        payload = ExecutionFailurePayloadV1(
            failure_kind=validated.failure_kind,
            safe_summary=validated.safe_summary,
            failure_code=validated.failure_code,
        )
        catalog_entry = get_catalog_entry(RuntimeEventType.EXECUTION_FAILED)
        if catalog_entry is None:
            raise RuntimeError("EXECUTION_FAILED missing event catalog entry")
        event = RuntimeEvent(
            tenant_id=validated.tenant_id,
            task_id=validated.task_id,
            run_id=validated.run_id,
            attempt_id=validated.attempt_id,
            execution_id=validated.execution_id,
            event_type=RuntimeEventType.EXECUTION_FAILED,
            phase=catalog_entry.phase,
            severity=EventSeverity.ERROR,
            payload={},
        )
        event = runtime_event_with_payload(event, payload)
        try:
            self._bus.record(event, tenant_id=validated.tenant_id)
        except MandatoryEvidencePersistenceError:
            return ExecutionFailureEvidenceRecordResult(
                status=ExecutionFailureEvidenceRecordStatus.UNAVAILABLE,
                event_id=None,
            )
        return ExecutionFailureEvidenceRecordResult(
            status=ExecutionFailureEvidenceRecordStatus.PERSISTED,
            event_id=event.event_id,
        )


__all__ = ["RuntimeEventExecutionFailureEvidenceRecorder"]

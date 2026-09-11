# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""RuntimeEvent bus adapter for external operation failures (R2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    ExecutionId,
    RunId,
    TaskId,
)
from intergrax.contracts.external_operations.failure import ExternalOperationFailureKind
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_catalog import get_catalog_entry
from intergrax.runtime.events.payload_registry import runtime_event_with_payload
from intergrax.runtime.events.payloads.canonical import ExternalOperationFailurePayloadV1
from intergrax.runtime.events.persistence_contract import MandatoryEvidencePersistenceError
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType


class ExternalOperationFailureRecordStatus(StrEnum):
    PERSISTED = "persisted"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ExternalOperationFailureRecordResult:
    status: ExternalOperationFailureRecordStatus
    event_id: EventId | None = None

    @staticmethod
    def persisted(event_id: EventId) -> ExternalOperationFailureRecordResult:
        return ExternalOperationFailureRecordResult(
            status=ExternalOperationFailureRecordStatus.PERSISTED,
            event_id=event_id,
        )

    @staticmethod
    def unavailable() -> ExternalOperationFailureRecordResult:
        return ExternalOperationFailureRecordResult(
            status=ExternalOperationFailureRecordStatus.UNAVAILABLE,
        )


class RuntimeEventExternalOperationFailureRecorder:
    __slots__ = ("_bus",)

    def __init__(self, bus: RuntimeEventBus) -> None:
        self._bus = bus

    def record_failure(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        attempt_id: AttemptId,
        execution_id: ExecutionId,
        operation_attempt_id: str,
        provider_id: str,
        operation_type: str,
        failure_kind: ExternalOperationFailureKind,
        retryable: bool,
        evidence_refs: tuple[str, ...] = (),
    ) -> ExternalOperationFailureRecordResult:
        if self._bus.persistence is None:
            return ExternalOperationFailureRecordResult.unavailable()
        payload = ExternalOperationFailurePayloadV1(
            execution_id=execution_id,
            operation_attempt_id=operation_attempt_id,
            provider_id=provider_id,
            operation_type=operation_type,
            failure_kind=failure_kind,
            retryable=retryable,
            evidence_refs=evidence_refs,
        )
        catalog_entry = get_catalog_entry(RuntimeEventType.EXTERNAL_OPERATION_FAILED)
        if catalog_entry is None:
            raise RuntimeError("EXTERNAL_OPERATION_FAILED missing event catalog entry")
        event = RuntimeEvent(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            event_type=RuntimeEventType.EXTERNAL_OPERATION_FAILED,
            phase=catalog_entry.phase,
            severity=EventSeverity.ERROR,
            payload={},
        )
        event = runtime_event_with_payload(event, payload)
        try:
            self._bus.record(event, tenant_id=tenant_id)
        except MandatoryEvidencePersistenceError:
            return ExternalOperationFailureRecordResult.unavailable()
        return ExternalOperationFailureRecordResult.persisted(event.event_id)


__all__ = [
    "ExternalOperationFailureRecordResult",
    "ExternalOperationFailureRecordStatus",
    "RuntimeEventExternalOperationFailureRecorder",
]

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-memory EE terminal outcome store and CanonicalExecutionOutcomeReader adapter."""

from __future__ import annotations

import threading

from intergrax.autonomous_work.recovery_orchestration_ports import (
    CanonicalExecutionOutcomeReader,
    CanonicalExecutionTerminalDisposition,
    CanonicalExecutionTerminalOutcome,
)
from intergrax.contracts.execution.execution_terminal_outcome_by_execution_id import (
    ExecutionTerminalOutcomeByExecutionIdDisposition,
    ExecutionTerminalOutcomeByExecutionIdStore,
)
from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id


class InMemoryExecutionTerminalOutcomeByExecutionIdStore(
    ExecutionTerminalOutcomeByExecutionIdStore,
):
    """Process-local terminal outcome index — plugin replacement for durable EE stores."""

    __slots__ = ("_lock", "_records")

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, ExecutionTerminalOutcomeByExecutionIdDisposition] = {}

    def get_recorded_disposition(
        self,
        execution_id: ExecutionId,
    ) -> ExecutionTerminalOutcomeByExecutionIdDisposition | None:
        key = str(validate_execution_id(execution_id))
        with self._lock:
            return self._records.get(key)

    def record_terminal_disposition(
        self,
        execution_id: ExecutionId,
        disposition: ExecutionTerminalOutcomeByExecutionIdDisposition,
    ) -> None:
        key = str(validate_execution_id(execution_id))
        with self._lock:
            existing = self._records.get(key)
            if existing is not None and existing != disposition:
                raise ValueError(
                    "execution terminal outcome conflict for execution_id",
                )
            self._records[key] = disposition


class ExecutionTerminalOutcomeByExecutionIdCanonicalReader(
    CanonicalExecutionOutcomeReader
):
    """Maps EE terminal outcome records to worker recovery semantics."""

    __slots__ = ("_read_port",)

    def __init__(
        self,
        read_port: ExecutionTerminalOutcomeByExecutionIdStore,
    ) -> None:
        self._read_port = read_port

    def get_terminal_outcome(
        self,
        execution_id: ExecutionId,
    ) -> CanonicalExecutionTerminalOutcome:
        normalized = validate_execution_id(execution_id)
        recorded = self._read_port.get_recorded_disposition(normalized)
        if recorded is None:
            return CanonicalExecutionTerminalOutcome(
                disposition=CanonicalExecutionTerminalDisposition.IN_PROGRESS,
                execution_id=normalized,
            )
        if recorded is ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED:
            disposition = CanonicalExecutionTerminalDisposition.SUCCEEDED
        else:
            disposition = CanonicalExecutionTerminalDisposition.FAILED
        return CanonicalExecutionTerminalOutcome(
            disposition=disposition,
            execution_id=normalized,
        )


def build_canonical_execution_outcome_reader(
    store: ExecutionTerminalOutcomeByExecutionIdStore,
) -> CanonicalExecutionOutcomeReader:
    return ExecutionTerminalOutcomeByExecutionIdCanonicalReader(store)


def record_delegate_terminal_disposition(
    store: ExecutionTerminalOutcomeByExecutionIdStore | None,
    *,
    execution_id: ExecutionId,
    disposition: object,
) -> None:
    if store is None:
        return
    from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
        QualifiedCapabilityExecutionDispatchDisposition,
    )

    if disposition is QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED:
        store.record_terminal_disposition(
            execution_id,
            ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED,
        )
    elif disposition in {
        QualifiedCapabilityExecutionDispatchDisposition.FAILED,
        QualifiedCapabilityExecutionDispatchDisposition.REJECTED,
        QualifiedCapabilityExecutionDispatchDisposition.UNAVAILABLE,
    }:
        store.record_terminal_disposition(
            execution_id,
            ExecutionTerminalOutcomeByExecutionIdDisposition.FAILED,
        )


__all__ = [
    "ExecutionTerminalOutcomeByExecutionIdCanonicalReader",
    "InMemoryExecutionTerminalOutcomeByExecutionIdStore",
    "build_canonical_execution_outcome_reader",
    "record_delegate_terminal_disposition",
]

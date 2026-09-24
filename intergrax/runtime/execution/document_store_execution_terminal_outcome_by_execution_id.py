# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ConditionalDocumentStore-backed EE terminal outcome index (UCA-6C-R6-R5.9)."""

from __future__ import annotations

import threading

from intergrax.contracts.execution.execution_terminal_outcome_by_execution_id import (
    ExecutionTerminalOutcomeByExecutionIdDisposition,
    ExecutionTerminalOutcomeByExecutionIdStore,
)
from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
)

_PARTITION = "execution.terminal_outcome_by_execution_id"
_SCHEMA = "execution_terminal_outcome.v1"


class DocumentStoreExecutionTerminalOutcomeByExecutionIdStore(
    ExecutionTerminalOutcomeByExecutionIdStore,
):
    """Durable multi-host-safe terminal outcome store — one immutable disposition per ExecutionId."""

    __slots__ = ("_document_store", "_lock")

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "execution terminal outcome persistence requires ConditionalDocumentStore",
            )
        self._document_store = document_store
        self._lock = threading.RLock()

    def get_recorded_disposition(
        self,
        execution_id: ExecutionId,
    ) -> ExecutionTerminalOutcomeByExecutionIdDisposition | None:
        key = str(validate_execution_id(execution_id))
        with self._lock:
            record = self._document_store.get(_PARTITION, key)
        if record is None:
            return None
        raw = record.data.get("disposition")
        if raw == ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED.value:
            return ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED
        if raw == ExecutionTerminalOutcomeByExecutionIdDisposition.FAILED.value:
            return ExecutionTerminalOutcomeByExecutionIdDisposition.FAILED
        raise RuntimeError("corrupt execution terminal outcome record")

    def record_terminal_disposition(
        self,
        execution_id: ExecutionId,
        disposition: ExecutionTerminalOutcomeByExecutionIdDisposition,
    ) -> None:
        key = str(validate_execution_id(execution_id))
        document = DocumentRecord(
            partition_key=_PARTITION,
            row_key=key,
            data={
                "schema": _SCHEMA,
                "disposition": disposition.value,
            },
        )
        with self._lock:
            inserted = self._document_store.put_if_absent(document)
            if inserted:
                return
            existing = self.get_recorded_disposition(execution_id)
            if existing is None:
                raise RuntimeError(
                    "terminal outcome record missing after insert conflict"
                )
            if existing != disposition:
                raise ValueError(
                    "execution terminal outcome conflict for execution_id",
                )


__all__ = ["DocumentStoreExecutionTerminalOutcomeByExecutionIdStore"]

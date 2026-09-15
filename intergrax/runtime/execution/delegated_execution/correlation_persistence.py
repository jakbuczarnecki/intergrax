# © Artur Czarnecki. All rights reserved.

"""Delegated invocation correlation persistence adapters (P2.1-S2C)."""

from __future__ import annotations

import json
import threading
from pydantic import ValidationError

from intergrax.contracts.delegated_invocation_correlation import (
    DelegatedInvocationCorrelationConflictError,
    DelegatedInvocationCorrelationIntegrityError,
    DelegatedInvocationCorrelationPersistenceError,
    DelegatedInvocationCorrelationRecord,
    DelegatedInvocationCorrelationStore,
    correlation_records_equivalent,
)
from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
)

_DOCUMENT_PARTITION = "intergrax.delegated_invocation_correlation.v1"


def encode_correlation_record(record: DelegatedInvocationCorrelationRecord) -> bytes:
    return record.model_dump_json().encode("utf-8")


def decode_correlation_record(raw: bytes) -> DelegatedInvocationCorrelationRecord:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DelegatedInvocationCorrelationIntegrityError(
            "correlation record is not valid JSON",
        ) from exc
    try:
        return DelegatedInvocationCorrelationRecord.model_validate(payload)
    except ValidationError as exc:
        raise DelegatedInvocationCorrelationIntegrityError(
            "correlation record failed schema validation",
        ) from exc


class InMemoryDelegatedInvocationCorrelationStore(DelegatedInvocationCorrelationStore):
    """Shared-backend in-memory store for tests and single-process hosts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, DelegatedInvocationCorrelationRecord] = {}

    @property
    def is_durable(self) -> bool:
        return False

    def persist(self, record: DelegatedInvocationCorrelationRecord) -> None:
        key = str(record.binding.execution_id)
        with self._lock:
            existing = self._records.get(key)
            if existing is None:
                self._records[key] = record
                return
            if correlation_records_equivalent(existing, record):
                return
            raise DelegatedInvocationCorrelationConflictError(
                "delegated invocation correlation conflict for execution_id",
            )

    def get_by_execution_id(
        self,
        execution_id: ExecutionId,
    ) -> DelegatedInvocationCorrelationRecord | None:
        key = str(validate_execution_id(execution_id))
        with self._lock:
            return self._records.get(key)


class DocumentStoreDelegatedInvocationCorrelationStore(
    DelegatedInvocationCorrelationStore,
):
    """ConditionalDocumentStore-backed delegated invocation correlation."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "delegated invocation correlation requires ConditionalDocumentStore",
            )
        self._document_store = document_store

    @property
    def is_durable(self) -> bool:
        return True

    def persist(self, record: DelegatedInvocationCorrelationRecord) -> None:
        row_key = str(record.binding.execution_id)
        existing_doc = self._document_store.get(_DOCUMENT_PARTITION, row_key)
        if existing_doc is not None:
            existing = _document_to_correlation(existing_doc)
            if correlation_records_equivalent(existing, record):
                return
            raise DelegatedInvocationCorrelationConflictError(
                "delegated invocation correlation conflict for execution_id",
            )
        document = DocumentRecord(
            partition_key=_DOCUMENT_PARTITION,
            row_key=row_key,
            data={"correlation": encode_correlation_record(record).decode("utf-8")},
        )
        if not self._document_store.put_if_absent(document):
            stored = self.get_by_execution_id(record.binding.execution_id)
            if stored is not None and correlation_records_equivalent(stored, record):
                return
            raise DelegatedInvocationCorrelationConflictError(
                "delegated invocation correlation conflict for execution_id",
            )

    def get_by_execution_id(
        self,
        execution_id: ExecutionId,
    ) -> DelegatedInvocationCorrelationRecord | None:
        row_key = str(validate_execution_id(execution_id))
        document = self._document_store.get(_DOCUMENT_PARTITION, row_key)
        if document is None:
            return None
        return _document_to_correlation(document)


def _document_to_correlation(document: DocumentRecord) -> DelegatedInvocationCorrelationRecord:
    raw = document.data.get("correlation")
    if not isinstance(raw, str):
        raise DelegatedInvocationCorrelationIntegrityError(
            "invalid delegated invocation correlation document",
        )
    return decode_correlation_record(raw.encode("utf-8"))


def wire_delegated_invocation_correlation_store(
    *,
    document_store: ConditionalDocumentStore | None = None,
    in_memory_backend: InMemoryDelegatedInvocationCorrelationStore | None = None,
) -> DelegatedInvocationCorrelationStore:
    """Composition helper for correlation store wiring."""
    if document_store is not None:
        return DocumentStoreDelegatedInvocationCorrelationStore(document_store)
    if in_memory_backend is not None:
        return in_memory_backend
    return InMemoryDelegatedInvocationCorrelationStore()


__all__ = [
    "DocumentStoreDelegatedInvocationCorrelationStore",
    "InMemoryDelegatedInvocationCorrelationStore",
    "decode_correlation_record",
    "encode_correlation_record",
    "wire_delegated_invocation_correlation_store",
]

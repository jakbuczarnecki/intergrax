# © Artur Czarnecki. All rights reserved.

"""Provider-neutral durable execution deadline authority persistence."""

from __future__ import annotations

import threading

from intergrax.contracts.execution_identity import RunId, validate_run_id
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)

_KV_KEY_PREFIX = "run_execution_deadline"
_DOCUMENT_STORE_PARTITION_PREFIX = "intergrax.run_execution_deadline.v1"


class ExecutionDeadlineCodecError(RuntimeError):
    """Raised when deadline authority bytes cannot be decoded safely."""


def _kv_storage_key(run_id: RunId) -> str:
    return f"{_KV_KEY_PREFIX}:{run_id}"


def _document_partition(tenant_id: str) -> str:
    return f"{_DOCUMENT_STORE_PARTITION_PREFIX}:{tenant_id}"


def _document_row_key(run_id: RunId) -> str:
    return str(run_id)


class InMemoryExecutionDeadlinePersistence:
    """Process-local CAS store for tests and single-process hosts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[tuple[str, str], bytes] = {}

    def load(self, *, tenant_id: str, run_id: RunId) -> bytes | None:
        key = (tenant_id, str(run_id))
        with self._lock:
            return self._records.get(key)

    def compare_and_create(
        self,
        *,
        tenant_id: str,
        run_id: RunId,
        expected: bytes | None,
        encoded_snapshot: bytes,
    ) -> bool:
        key = (tenant_id, str(validate_run_id(run_id)))
        with self._lock:
            current = self._records.get(key)
            if expected is None and current is not None:
                return False
            if expected is not None and current != expected:
                return False
            self._records[key] = encoded_snapshot
            return True


class KvExecutionDeadlinePersistence:
    def __init__(self, kv_store: DistributedKVStore) -> None:
        self._kv_store = kv_store

    def load(self, *, tenant_id: str, run_id: RunId) -> bytes | None:
        return self._kv_store.get(tenant_id=tenant_id, key=_kv_storage_key(run_id))

    def compare_and_create(
        self,
        *,
        tenant_id: str,
        run_id: RunId,
        expected: bytes | None,
        encoded_snapshot: bytes,
    ) -> bool:
        return self._kv_store.compare_and_set(
            tenant_id=tenant_id,
            key=_kv_storage_key(run_id),
            expected=expected,
            new_value=encoded_snapshot,
        )


class DocumentStoreExecutionDeadlinePersistence:
    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        self._document_store = document_store

    def load(self, *, tenant_id: str, run_id: RunId) -> bytes | None:
        partition = _document_partition(tenant_id)
        record = self._document_store.get(partition, _document_row_key(run_id))
        if record is None:
            return None
        encoded = record.data.get("snapshot")
        if encoded is None:
            return None
        if not isinstance(encoded, str):
            raise ExecutionDeadlineCodecError(
                "execution deadline document snapshot must be a string",
            )
        return encoded.encode("utf-8")

    def compare_and_create(
        self,
        *,
        tenant_id: str,
        run_id: RunId,
        expected: bytes | None,
        encoded_snapshot: bytes,
    ) -> bool:
        partition_key = _document_partition(tenant_id)
        row_key = _document_row_key(run_id)
        replacement = DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data={"snapshot": encoded_snapshot.decode("utf-8")},
        )
        if expected is None:
            return self._document_store.put_if_absent(replacement)
        existing = self._document_store.get(partition_key, row_key)
        if existing is None:
            return False
        current = self._snapshot_bytes_from_record(existing)
        if current != expected:
            return False
        return self._document_store.replace_if_match(
            expected=existing,
            replacement=replacement,
        )

    @staticmethod
    def _snapshot_bytes_from_record(record: DocumentRecord) -> bytes:
        encoded = record.data.get("snapshot")
        if not isinstance(encoded, str):
            raise ExecutionDeadlineCodecError(
                "execution deadline document snapshot must be a string",
            )
        return encoded.encode("utf-8")


def wire_execution_deadline_persistence(
    *,
    kv_store: DistributedKVStore | None = None,
    document_store: DocumentStore | None = None,
) -> KvExecutionDeadlinePersistence | DocumentStoreExecutionDeadlinePersistence:
    if kv_store is not None and document_store is not None:
        raise ValueError(
            "wire_execution_deadline_persistence accepts kv_store or document_store, not both",
        )
    if kv_store is not None:
        return KvExecutionDeadlinePersistence(kv_store)
    if document_store is not None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "execution deadline persistence requires ConditionalDocumentStore",
            )
        return DocumentStoreExecutionDeadlinePersistence(document_store)
    raise ValueError(
        "wire_execution_deadline_persistence requires kv_store or document_store",
    )

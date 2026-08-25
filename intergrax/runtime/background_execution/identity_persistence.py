# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable transport→canonical identity resolution (BG-EXEC-2)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from intergrax.contracts.execution_identity import (
    RunId,
    TaskId,
    mint_run_id,
    mint_task_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)
from intergrax.runtime.background_execution.transport_ref import (
    BackgroundTransportExecutionRef,
)

_IDENTITY_RECORD_SEPARATOR = "\n"
_KV_KEY_PREFIX = "bg_exec_identity"
_DOCUMENT_STORE_PARTITION_PREFIX = "intergrax.bg_exec_identity.v1"


def _encode_identity_record(*, task_id: TaskId, run_id: RunId) -> bytes:
    return f"{task_id}{_IDENTITY_RECORD_SEPARATOR}{run_id}".encode("utf-8")


def _decode_identity_record(raw: bytes) -> tuple[TaskId, RunId]:
    try:
        task_raw, run_raw = raw.decode("utf-8").split(_IDENTITY_RECORD_SEPARATOR, 1)
    except ValueError as exc:
        raise RuntimeError("invalid background execution identity record") from exc
    return validate_task_id(task_raw), validate_run_id(run_raw)


def _kv_storage_key(transport_ref: BackgroundTransportExecutionRef) -> str:
    return f"{_KV_KEY_PREFIX}:{transport_ref.provider}:{transport_ref.transport_task_id}"


def _document_partition(tenant_id: str) -> str:
    return f"{_DOCUMENT_STORE_PARTITION_PREFIX}:{tenant_id}"


def _document_row_key(transport_ref: BackgroundTransportExecutionRef) -> str:
    return f"{transport_ref.provider}:{transport_ref.transport_task_id}"


class BackgroundExecutionIdentityPersistence(ABC):
    """Platform-owned durable mapping from transport identity to canonical TaskId/RunId."""

    @abstractmethod
    def resolve_or_create(
        self,
        transport_ref: BackgroundTransportExecutionRef,
    ) -> tuple[TaskId, RunId]:
        """Return stable canonical identity for one transport execution."""


class KvBackgroundExecutionIdentityPersistence(BackgroundExecutionIdentityPersistence):
    """DistributedKVStore-backed execution identity registry."""

    def __init__(self, kv_store: DistributedKVStore) -> None:
        self._kv_store = kv_store

    def resolve_or_create(
        self,
        transport_ref: BackgroundTransportExecutionRef,
    ) -> tuple[TaskId, RunId]:
        key = _kv_storage_key(transport_ref)
        existing = self._kv_store.get(
            tenant_id=transport_ref.tenant_id,
            key=key,
        )
        if existing is not None:
            return _decode_identity_record(existing)

        task_id = mint_task_id()
        run_id = mint_run_id()
        encoded = _encode_identity_record(task_id=task_id, run_id=run_id)
        if self._kv_store.compare_and_set(
            tenant_id=transport_ref.tenant_id,
            key=key,
            expected=None,
            new_value=encoded,
        ):
            return task_id, run_id

        raced = self._kv_store.get(tenant_id=transport_ref.tenant_id, key=key)
        if raced is None:
            raise RuntimeError("background execution identity resolution failed")
        return _decode_identity_record(raced)


class DocumentStoreBackgroundExecutionIdentityPersistence(
    BackgroundExecutionIdentityPersistence,
):
    """ConditionalDocumentStore-backed execution identity registry."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "background execution identity persistence requires ConditionalDocumentStore"
            )
        self._document_store = document_store

    def resolve_or_create(
        self,
        transport_ref: BackgroundTransportExecutionRef,
    ) -> tuple[TaskId, RunId]:
        partition_key = _document_partition(transport_ref.tenant_id)
        row_key = _document_row_key(transport_ref)
        existing = self._document_store.get(partition_key, row_key)
        if existing is not None:
            return self._record_to_identity(existing)

        task_id = mint_task_id()
        run_id = mint_run_id()
        document = DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data={
                "task_id": str(task_id),
                "run_id": str(run_id),
            },
        )
        if self._document_store.put_if_absent(document):
            return task_id, run_id

        raced = self._document_store.get(partition_key, row_key)
        if raced is None:
            raise RuntimeError("background execution identity resolution failed")
        return self._record_to_identity(raced)

    @staticmethod
    def _record_to_identity(record: DocumentRecord) -> tuple[TaskId, RunId]:
        task_raw = record.data.get("task_id")
        run_raw = record.data.get("run_id")
        if not isinstance(task_raw, str) or not isinstance(run_raw, str):
            raise RuntimeError("invalid background execution identity record")
        return validate_task_id(task_raw), validate_run_id(run_raw)


def wire_background_execution_identity_persistence(
    *,
    kv_store: DistributedKVStore | None = None,
    document_store: DocumentStore | None = None,
) -> BackgroundExecutionIdentityPersistence:
    """Platform composition boundary: storage capability → identity persistence."""
    if kv_store is not None and document_store is not None:
        raise ValueError(
            "wire_background_execution_identity_persistence accepts kv_store or "
            "document_store, not both",
        )
    if kv_store is not None:
        return KvBackgroundExecutionIdentityPersistence(kv_store)
    if document_store is not None:
        return DocumentStoreBackgroundExecutionIdentityPersistence(document_store)
    raise ValueError(
        "wire_background_execution_identity_persistence requires kv_store or document_store",
    )

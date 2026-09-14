# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable transport→canonical identity resolution (BG-EXEC-2)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)
from intergrax.contracts.npsc5f_compatibility import (
    BackgroundExecutionIdentityConflictError,
)
from intergrax.runtime.background_execution.identity_dual_read import (
    reconcile_dual_read_records,
)
from intergrax.runtime.background_execution.identity_record_codec import (
    BG_EXEC_IDENTITY_DOCUMENT_PARTITION_V1,
    BG_EXEC_IDENTITY_DOCUMENT_PARTITION_V2,
    InvalidBackgroundExecutionIdentityV1RecordError,
    InvalidBackgroundExecutionIdentityV2RecordError,
    decode_background_identity_kv_record,
    decode_document_identity_v1_record,
    decode_document_identity_v2_record,
    encode_background_identity_v2_record,
    same_identity_triplet,
)
from intergrax.runtime.background_execution.identity_types import (
    PersistedBackgroundExecutionIdentity,
)
from intergrax.runtime.background_execution.transport_ref import (
    BackgroundTransportExecutionRef,
)
from intergrax.runtime.execution.identity_authority import BackgroundTransportIdentity
from intergrax.runtime.observability.causal_evidence_enrichment import (
    CanonicalExecutionIdLookupPort,
)

_KV_KEY_PREFIX = "bg_exec_identity"


def _kv_storage_key(transport_ref: BackgroundTransportExecutionRef) -> str:
    return (
        f"{_KV_KEY_PREFIX}:{transport_ref.provider}:{transport_ref.transport_task_id}"
    )


def _document_partition_v2(tenant_id: str) -> str:
    return f"{BG_EXEC_IDENTITY_DOCUMENT_PARTITION_V2}:{tenant_id}"


def _document_partition_v1(tenant_id: str) -> str:
    return f"{BG_EXEC_IDENTITY_DOCUMENT_PARTITION_V1}:{tenant_id}"


def _document_row_key(transport_ref: BackgroundTransportExecutionRef) -> str:
    return f"{transport_ref.provider}:{transport_ref.transport_task_id}"


class BackgroundExecutionIdentityPersistence(ABC):
    """Platform-owned durable mapping from transport identity to canonical identity."""

    @abstractmethod
    def load(
        self,
        transport_ref: BackgroundTransportExecutionRef,
    ) -> PersistedBackgroundExecutionIdentity | None:
        """Return persisted canonical identity for one transport execution."""

    @abstractmethod
    def store_if_absent(
        self,
        transport_ref: BackgroundTransportExecutionRef,
        identity: BackgroundTransportIdentity,
    ) -> PersistedBackgroundExecutionIdentity:
        """Persist canonical identity when no durable mapping exists yet."""


class KvBackgroundExecutionIdentityPersistence(BackgroundExecutionIdentityPersistence):
    """DistributedKVStore-backed execution identity registry."""

    def __init__(
        self,
        kv_store: DistributedKVStore,
        *,
        legacy_lookup: CanonicalExecutionIdLookupPort | None = None,
    ) -> None:
        self._kv_store = kv_store
        self._legacy_lookup = legacy_lookup

    def load(
        self,
        transport_ref: BackgroundTransportExecutionRef,
    ) -> PersistedBackgroundExecutionIdentity | None:
        key = _kv_storage_key(transport_ref)
        existing = self._kv_store.get(
            tenant_id=transport_ref.tenant_id,
            key=key,
        )
        if existing is None:
            return None
        try:
            decoded = decode_background_identity_kv_record(existing)
        except ValueError as exc:
            raise RuntimeError("invalid background execution identity record") from exc
        if decoded.kind == "complete_v2":
            return reconcile_dual_read_records(
                v2_candidate=decoded,
                v1_candidate=None,
                tenant_id=transport_ref.tenant_id,
                lookup=self._legacy_lookup,
            )
        return reconcile_dual_read_records(
            v2_candidate=None,
            v1_candidate=decoded,
            tenant_id=transport_ref.tenant_id,
            lookup=self._legacy_lookup,
        )

    def store_if_absent(
        self,
        transport_ref: BackgroundTransportExecutionRef,
        identity: BackgroundTransportIdentity,
    ) -> PersistedBackgroundExecutionIdentity:
        key = _kv_storage_key(transport_ref)
        encoded = encode_background_identity_v2_record(
            task_id=identity.task_id,
            run_id=identity.run_id,
            attempt_id=identity.attempt_id,
            execution_id=identity.execution_id,
        )
        if self._kv_store.compare_and_set(
            tenant_id=transport_ref.tenant_id,
            key=key,
            expected=None,
            new_value=encoded,
        ):
            return PersistedBackgroundExecutionIdentity(
                task_id=identity.task_id,
                run_id=identity.run_id,
                attempt_id=identity.attempt_id,
                execution_id=identity.execution_id,
            )

        raced = self._kv_store.get(tenant_id=transport_ref.tenant_id, key=key)
        if raced is None:
            raise RuntimeError("background execution identity resolution failed")
        loaded = self.load(transport_ref)
        if loaded is None:
            raise RuntimeError("background execution identity resolution failed")
        return loaded


class DocumentStoreBackgroundExecutionIdentityPersistence(
    BackgroundExecutionIdentityPersistence,
):
    """ConditionalDocumentStore-backed execution identity registry."""

    def __init__(
        self,
        document_store: ConditionalDocumentStore,
        *,
        legacy_lookup: CanonicalExecutionIdLookupPort | None = None,
    ) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "background execution identity persistence requires ConditionalDocumentStore"
            )
        self._document_store = document_store
        self._legacy_lookup = legacy_lookup

    def load(
        self,
        transport_ref: BackgroundTransportExecutionRef,
    ) -> PersistedBackgroundExecutionIdentity | None:
        row_key = _document_row_key(transport_ref)
        v2_partition = _document_partition_v2(transport_ref.tenant_id)
        v1_partition = _document_partition_v1(transport_ref.tenant_id)
        v2_record = self._document_store.get(v2_partition, row_key)
        v1_record = self._document_store.get(v1_partition, row_key)
        v2_candidate = None
        v1_candidate = None
        if v2_record is not None:
            try:
                v2_candidate = decode_document_identity_v2_record(dict(v2_record.data))
            except InvalidBackgroundExecutionIdentityV2RecordError:
                raise
            except ValueError as exc:
                raise RuntimeError(
                    "invalid background execution identity record"
                ) from exc
        if v1_record is not None:
            try:
                decoded_v1 = decode_document_identity_v1_record(dict(v1_record.data))
            except InvalidBackgroundExecutionIdentityV1RecordError:
                raise
            except ValueError as exc:
                raise RuntimeError(
                    "invalid background execution identity record"
                ) from exc
            if v1_candidate is None:
                v1_candidate = decoded_v1
            elif not same_identity_triplet(v1_candidate, decoded_v1):
                raise BackgroundExecutionIdentityConflictError(
                    "conflicting v1 background identity records",
                )
        if v2_candidate is None and v1_candidate is None:
            return None
        return reconcile_dual_read_records(
            v2_candidate=v2_candidate,
            v1_candidate=v1_candidate,
            tenant_id=transport_ref.tenant_id,
            lookup=self._legacy_lookup,
        )

    def store_if_absent(
        self,
        transport_ref: BackgroundTransportExecutionRef,
        identity: BackgroundTransportIdentity,
    ) -> PersistedBackgroundExecutionIdentity:
        partition_key = _document_partition_v2(transport_ref.tenant_id)
        row_key = _document_row_key(transport_ref)
        document = DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data={
                "task_id": str(identity.task_id),
                "run_id": str(identity.run_id),
                "attempt_id": str(identity.attempt_id),
                "execution_id": str(identity.execution_id),
            },
        )
        if self._document_store.put_if_absent(document):
            return PersistedBackgroundExecutionIdentity(
                task_id=identity.task_id,
                run_id=identity.run_id,
                attempt_id=identity.attempt_id,
                execution_id=identity.execution_id,
            )

        loaded = self.load(transport_ref)
        if loaded is None:
            raise RuntimeError("background execution identity resolution failed")
        return loaded


def wire_background_execution_identity_persistence(
    *,
    kv_store: DistributedKVStore | None = None,
    document_store: DocumentStore | None = None,
    legacy_lookup: CanonicalExecutionIdLookupPort | None = None,
) -> BackgroundExecutionIdentityPersistence:
    """Platform composition boundary: storage capability → identity persistence."""
    if kv_store is not None and document_store is not None:
        raise ValueError(
            "wire_background_execution_identity_persistence accepts kv_store or "
            "document_store, not both",
        )
    if kv_store is not None:
        return KvBackgroundExecutionIdentityPersistence(
            kv_store,
            legacy_lookup=legacy_lookup,
        )
    if document_store is not None:
        return DocumentStoreBackgroundExecutionIdentityPersistence(
            document_store,
            legacy_lookup=legacy_lookup,
        )
    raise ValueError(
        "wire_background_execution_identity_persistence requires kv_store or document_store",
    )


__all__ = [
    "BackgroundExecutionIdentityPersistence",
    "DocumentStoreBackgroundExecutionIdentityPersistence",
    "KvBackgroundExecutionIdentityPersistence",
    "PersistedBackgroundExecutionIdentity",
    "wire_background_execution_identity_persistence",
]

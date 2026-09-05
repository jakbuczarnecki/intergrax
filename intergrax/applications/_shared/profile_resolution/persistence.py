# © Artur Czarnecki. All rights reserved.

"""Provider-backed effective profile revision and pinning persistence (P1.2A)."""

from __future__ import annotations

import json
from typing import Any

from intergrax.applications.contracts.profile_resolution.errors import (
    EffectiveProfileRevisionConflictError,
    EffectiveProfileRevisionError,
)
from intergrax.applications.contracts.profile_resolution.execution_binding import (
    EffectiveProfileExecutionBinding,
)
from intergrax.applications.contracts.profile_resolution.revision import (
    EffectiveProfileRevision,
    EffectiveProfileRevisionScope,
)
from intergrax.applications.contracts.profile_resolution.revision_id import (
    EffectiveProfileRevisionId,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)

_REVISION_KV_PREFIX = "effective_profile_revision"
_PINNING_KV_PREFIX = "effective_profile_execution_pinning"
_REVISION_DOCUMENT_PARTITION_PREFIX = "intergrax.effective_profile_revision.v1"
_PINNING_DOCUMENT_PARTITION_PREFIX = "intergrax.effective_profile_execution_pinning.v1"
_REVISION_SCHEMA_VERSION = 1
_PINNING_SCHEMA_VERSION = 1


def _revision_scope_key(scope: EffectiveProfileRevisionScope) -> tuple[str, str | None]:
    return (scope.application_id, scope.tenant_id)


def _revision_kv_key(revision_id: EffectiveProfileRevisionId) -> str:
    return f"{_REVISION_KV_PREFIX}:{revision_id.value}"


def _revision_document_partition(scope: EffectiveProfileRevisionScope) -> str:
    app_id, tenant_id = _revision_scope_key(scope)
    tenant_suffix = tenant_id or "_global"
    return f"{_REVISION_DOCUMENT_PARTITION_PREFIX}:{app_id}:{tenant_suffix}"


def _pinning_kv_key(execution_id: str) -> str:
    return f"{_PINNING_KV_PREFIX}:{execution_id}"


def _pinning_document_partition(tenant_id: str) -> str:
    return f"{_PINNING_DOCUMENT_PARTITION_PREFIX}:{tenant_id}"


def _revision_storage_payload(revision: EffectiveProfileRevision) -> dict[str, Any]:
    payload = json.loads(revision.model_dump_json())
    payload["revision_id"] = revision.revision_id.value
    if revision.predecessor_revision_id is not None:
        payload["predecessor_revision_id"] = revision.predecessor_revision_id.value
    return payload


def _binding_storage_payload(binding: EffectiveProfileExecutionBinding) -> dict[str, Any]:
    payload = json.loads(binding.model_dump_json())
    payload["execution_id"] = str(binding.execution_id)
    payload["revision_id"] = binding.revision_id.value
    return payload


def encode_effective_profile_revision(revision: EffectiveProfileRevision) -> bytes:
    payload: dict[str, Any] = {
        "schema_version": _REVISION_SCHEMA_VERSION,
        "revision": _revision_storage_payload(revision),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_effective_profile_revision(raw: bytes) -> EffectiveProfileRevision:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EffectiveProfileRevisionError("invalid effective profile revision encoding") from exc
    if not isinstance(payload, dict):
        raise EffectiveProfileRevisionError("invalid effective profile revision payload")
    if payload.get("schema_version") != _REVISION_SCHEMA_VERSION:
        raise EffectiveProfileRevisionError("unsupported effective profile revision schema version")
    revision_raw = payload.get("revision")
    if not isinstance(revision_raw, dict):
        raise EffectiveProfileRevisionError("invalid effective profile revision snapshot")
    return EffectiveProfileRevision.model_validate(revision_raw)


def encode_effective_profile_execution_binding(binding: EffectiveProfileExecutionBinding) -> bytes:
    payload: dict[str, Any] = {
        "schema_version": _PINNING_SCHEMA_VERSION,
        "binding": _binding_storage_payload(binding),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_effective_profile_execution_binding(raw: bytes) -> EffectiveProfileExecutionBinding:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EffectiveProfileRevisionError(
            "invalid effective profile execution binding encoding",
        ) from exc
    if not isinstance(payload, dict):
        raise EffectiveProfileRevisionError("invalid effective profile execution binding payload")
    if payload.get("schema_version") != _PINNING_SCHEMA_VERSION:
        raise EffectiveProfileRevisionError(
            "unsupported effective profile execution binding schema version",
        )
    binding_raw = payload.get("binding")
    if not isinstance(binding_raw, dict):
        raise EffectiveProfileRevisionError("invalid effective profile execution binding snapshot")
    return EffectiveProfileExecutionBinding.model_validate(binding_raw)


class KvEffectiveProfileRevisionStore:
    """DistributedKVStore-backed append-only revision store."""

    def __init__(self, kv_store: DistributedKVStore) -> None:
        self._kv_store = kv_store

    @property
    def is_durable(self) -> bool:
        return True

    def save(self, revision: EffectiveProfileRevision) -> None:
        encoded = encode_effective_profile_revision(revision)
        tenant_id = revision.scope.tenant_id or revision.scope.application_id
        if not self._kv_store.compare_and_set(
            tenant_id=tenant_id,
            key=_revision_kv_key(revision.revision_id),
            expected=None,
            new_value=encoded,
        ):
            raise EffectiveProfileRevisionConflictError(
                f"revision already exists: {revision.revision_id.value}",
            )

    def get(
        self,
        revision_id: EffectiveProfileRevisionId,
        *,
        scope: EffectiveProfileRevisionScope,
    ) -> EffectiveProfileRevision | None:
        tenant_id = scope.tenant_id or scope.application_id
        raw = self._kv_store.get(tenant_id=tenant_id, key=_revision_kv_key(revision_id))
        if raw is None:
            return None
        revision = decode_effective_profile_revision(raw)
        if revision.scope.application_id != scope.application_id:
            return None
        if scope.tenant_id is not None and revision.scope.tenant_id != scope.tenant_id:
            return None
        return revision


class DocumentStoreEffectiveProfileRevisionStore:
    """ConditionalDocumentStore-backed append-only revision store."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "effective profile revision persistence requires ConditionalDocumentStore",
            )
        self._document_store = document_store

    @property
    def is_durable(self) -> bool:
        return True

    def save(self, revision: EffectiveProfileRevision) -> None:
        partition_key = _revision_document_partition(revision.scope)
        document = DocumentRecord(
            partition_key=partition_key,
            row_key=revision.revision_id.value,
            data={"revision": encode_effective_profile_revision(revision).decode("utf-8")},
        )
        if not self._document_store.put_if_absent(document):
            raise EffectiveProfileRevisionConflictError(
                f"revision already exists: {revision.revision_id.value}",
            )

    def get(
        self,
        revision_id: EffectiveProfileRevisionId,
        *,
        scope: EffectiveProfileRevisionScope,
    ) -> EffectiveProfileRevision | None:
        record = self._document_store.get(
            _revision_document_partition(scope),
            revision_id.value,
        )
        if record is None:
            return None
        revision_raw = record.data.get("revision")
        if not isinstance(revision_raw, str):
            raise EffectiveProfileRevisionError("invalid effective profile revision document record")
        revision = decode_effective_profile_revision(revision_raw.encode("utf-8"))
        if revision.scope.application_id != scope.application_id:
            return None
        if scope.tenant_id is not None and revision.scope.tenant_id != scope.tenant_id:
            return None
        return revision


class KvEffectiveProfileExecutionPinningStore:
    """DistributedKVStore-backed execution revision pinning."""

    def __init__(self, kv_store: DistributedKVStore) -> None:
        self._kv_store = kv_store

    @property
    def is_durable(self) -> bool:
        return True

    def pin(self, binding: EffectiveProfileExecutionBinding) -> None:
        encoded = encode_effective_profile_execution_binding(binding)
        key = _pinning_kv_key(binding.execution_id)
        if self._kv_store.compare_and_set(
            tenant_id=binding.tenant_id,
            key=key,
            expected=None,
            new_value=encoded,
        ):
            return
        existing_raw = self._kv_store.get(tenant_id=binding.tenant_id, key=key)
        if existing_raw is None:
            raise EffectiveProfileRevisionError("execution pinning compare-and-set failed")
        existing = decode_effective_profile_execution_binding(existing_raw)
        if existing != binding:
            raise EffectiveProfileRevisionConflictError(
                f"execution already pinned: {binding.execution_id}",
            )

    def get(
        self,
        *,
        tenant_id: str,
        execution_id: str,
    ) -> EffectiveProfileExecutionBinding | None:
        raw = self._kv_store.get(
            tenant_id=tenant_id,
            key=_pinning_kv_key(execution_id),
        )
        if raw is None:
            return None
        return decode_effective_profile_execution_binding(raw)


class DocumentStoreEffectiveProfileExecutionPinningStore:
    """ConditionalDocumentStore-backed execution revision pinning."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "effective profile execution pinning requires ConditionalDocumentStore",
            )
        self._document_store = document_store

    @property
    def is_durable(self) -> bool:
        return True

    def pin(self, binding: EffectiveProfileExecutionBinding) -> None:
        partition_key = _pinning_document_partition(binding.tenant_id)
        row_key = str(binding.execution_id)
        document = DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data={"binding": encode_effective_profile_execution_binding(binding).decode("utf-8")},
        )
        if self._document_store.put_if_absent(document):
            return
        existing = self._document_store.get(partition_key, row_key)
        if existing is None:
            raise EffectiveProfileRevisionError("execution pinning document create failed")
        stored = decode_effective_profile_execution_binding(
            _binding_record_to_bytes(existing),
        )
        if stored != binding:
            raise EffectiveProfileRevisionConflictError(
                f"execution already pinned: {binding.execution_id}",
            )

    def get(
        self,
        *,
        tenant_id: str,
        execution_id: str,
    ) -> EffectiveProfileExecutionBinding | None:
        record = self._document_store.get(
            _pinning_document_partition(tenant_id),
            execution_id,
        )
        if record is None:
            return None
        return decode_effective_profile_execution_binding(_binding_record_to_bytes(record))


def _binding_record_to_bytes(record: DocumentRecord) -> bytes:
    binding = record.data.get("binding")
    if not isinstance(binding, str):
        raise EffectiveProfileRevisionError("invalid effective profile execution binding record")
    return binding.encode("utf-8")


def wire_effective_profile_revision_store(
    *,
    kv_store: DistributedKVStore | None = None,
    document_store: DocumentStore | None = None,
) -> KvEffectiveProfileRevisionStore | DocumentStoreEffectiveProfileRevisionStore:
    if kv_store is not None and document_store is not None:
        raise ValueError(
            "wire_effective_profile_revision_store accepts kv_store or document_store, not both",
        )
    if kv_store is not None:
        return KvEffectiveProfileRevisionStore(kv_store)
    if document_store is not None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "effective profile revision persistence requires ConditionalDocumentStore",
            )
        return DocumentStoreEffectiveProfileRevisionStore(document_store)
    raise ValueError(
        "wire_effective_profile_revision_store requires kv_store or document_store",
    )


def wire_effective_profile_execution_pinning_store(
    *,
    kv_store: DistributedKVStore | None = None,
    document_store: DocumentStore | None = None,
) -> KvEffectiveProfileExecutionPinningStore | DocumentStoreEffectiveProfileExecutionPinningStore:
    if kv_store is not None and document_store is not None:
        raise ValueError(
            "wire_effective_profile_execution_pinning_store accepts kv_store or "
            "document_store, not both",
        )
    if kv_store is not None:
        return KvEffectiveProfileExecutionPinningStore(kv_store)
    if document_store is not None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "effective profile execution pinning requires ConditionalDocumentStore",
            )
        return DocumentStoreEffectiveProfileExecutionPinningStore(document_store)
    raise ValueError(
        "wire_effective_profile_execution_pinning_store requires kv_store or document_store",
    )

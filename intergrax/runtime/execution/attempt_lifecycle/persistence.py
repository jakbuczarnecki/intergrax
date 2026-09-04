# © Artur Czarnecki. All rights reserved.

"""Provider-backed attempt lifecycle persistence (P0C-4)."""

from __future__ import annotations

import json
import threading
from typing import Any

from intergrax.contracts.attempt_lifecycle import (
    AttemptLifecycleError,
    AttemptLifecycleState,
    AttemptLifecycleStore,
    AttemptTransitionReason,
)
from intergrax.contracts.execution_identity import RunId, validate_attempt_id, validate_run_id
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)

_KV_KEY_PREFIX = "attempt_lifecycle"
_DOCUMENT_STORE_PARTITION_PREFIX = "intergrax.attempt_lifecycle.v1"
_SCHEMA_VERSION = 1


def _kv_storage_key(run_id: RunId) -> str:
    return f"{_KV_KEY_PREFIX}:{run_id}"


def _document_partition(tenant_id: str) -> str:
    return f"{_DOCUMENT_STORE_PARTITION_PREFIX}:{tenant_id}"


def _document_row_key(run_id: RunId) -> str:
    return str(run_id)


def encode_attempt_lifecycle_state(state: AttemptLifecycleState) -> bytes:
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "run_id": str(state.run_id),
        "active_attempt_id": str(state.active_attempt_id),
        "previous_attempt_id": (
            str(state.previous_attempt_id) if state.previous_attempt_id is not None else None
        ),
        "generation": state.generation,
        "transition_reason": (
            state.transition_reason.value if state.transition_reason is not None else None
        ),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_attempt_lifecycle_state(raw: bytes) -> AttemptLifecycleState:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttemptLifecycleError("invalid attempt lifecycle record encoding") from exc
    if not isinstance(payload, dict):
        raise AttemptLifecycleError("invalid attempt lifecycle record payload")
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise AttemptLifecycleError("unsupported attempt lifecycle schema version")
    return AttemptLifecycleState(
        run_id=validate_run_id(payload.get("run_id")),
        active_attempt_id=validate_attempt_id(payload.get("active_attempt_id")),
        previous_attempt_id=(
            validate_attempt_id(payload["previous_attempt_id"])
            if payload.get("previous_attempt_id") is not None
            else None
        ),
        generation=_require_generation(payload.get("generation")),
        transition_reason=_parse_transition_reason(payload.get("transition_reason")),
    )


def _require_generation(raw: object) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 1:
        raise AttemptLifecycleError("invalid attempt lifecycle generation")
    return raw


def _parse_transition_reason(raw: object) -> AttemptTransitionReason | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise AttemptLifecycleError("invalid attempt lifecycle transition reason")
    try:
        return AttemptTransitionReason(raw)
    except ValueError as exc:
        raise AttemptLifecycleError("invalid attempt lifecycle transition reason") from exc


class InMemoryAttemptLifecycleStore(AttemptLifecycleStore):
    """Process-local CAS store for tests and single-process hosts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[tuple[str, str], bytes] = {}

    @property
    def is_durable(self) -> bool:
        return False

    def load_raw(self, *, tenant_id: str, run_id: RunId) -> bytes | None:
        key = (tenant_id, str(run_id))
        with self._lock:
            return self._records.get(key)

    def compare_and_swap(
        self,
        *,
        tenant_id: str,
        run_id: RunId,
        expected: bytes | None,
        new_state: AttemptLifecycleState,
    ) -> bool:
        key = (tenant_id, str(run_id))
        encoded = encode_attempt_lifecycle_state(new_state)
        with self._lock:
            current = self._records.get(key)
            if current != expected:
                return False
            self._records[key] = encoded
            return True


class KvAttemptLifecycleStore(AttemptLifecycleStore):
    """DistributedKVStore-backed attempt lifecycle authority."""

    def __init__(self, kv_store: DistributedKVStore) -> None:
        self._kv_store = kv_store

    @property
    def is_durable(self) -> bool:
        return True

    def load_raw(self, *, tenant_id: str, run_id: RunId) -> bytes | None:
        return self._kv_store.get(tenant_id=tenant_id, key=_kv_storage_key(run_id))

    def compare_and_swap(
        self,
        *,
        tenant_id: str,
        run_id: RunId,
        expected: bytes | None,
        new_state: AttemptLifecycleState,
    ) -> bool:
        return self._kv_store.compare_and_set(
            tenant_id=tenant_id,
            key=_kv_storage_key(run_id),
            expected=expected,
            new_value=encode_attempt_lifecycle_state(new_state),
        )


class DocumentStoreAttemptLifecycleStore(AttemptLifecycleStore):
    """ConditionalDocumentStore-backed attempt lifecycle authority."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "attempt lifecycle persistence requires ConditionalDocumentStore",
            )
        self._document_store = document_store

    @property
    def is_durable(self) -> bool:
        return True

    def load_raw(self, *, tenant_id: str, run_id: RunId) -> bytes | None:
        record = self._document_store.get(_document_partition(tenant_id), _document_row_key(run_id))
        if record is None:
            return None
        return _record_to_bytes(record)

    def compare_and_swap(
        self,
        *,
        tenant_id: str,
        run_id: RunId,
        expected: bytes | None,
        new_state: AttemptLifecycleState,
    ) -> bool:
        partition_key = _document_partition(tenant_id)
        row_key = _document_row_key(run_id)
        encoded = encode_attempt_lifecycle_state(new_state)
        if expected is None:
            document = DocumentRecord(
                partition_key=partition_key,
                row_key=row_key,
                data={"lifecycle": encoded.decode("utf-8")},
            )
            return self._document_store.put_if_absent(document)
        existing = self._document_store.get(partition_key, row_key)
        if existing is None:
            return False
        current = _record_to_bytes(existing)
        if current != expected:
            return False
        replacement = DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data={"lifecycle": encoded.decode("utf-8")},
        )
        return self._document_store.replace_if_match(
            expected=existing,
            replacement=replacement,
        )


def _record_to_bytes(record: DocumentRecord) -> bytes:
    lifecycle = record.data.get("lifecycle")
    if not isinstance(lifecycle, str):
        raise AttemptLifecycleError("invalid attempt lifecycle document record")
    return lifecycle.encode("utf-8")


def wire_attempt_lifecycle_store(
    *,
    kv_store: DistributedKVStore | None = None,
    document_store: DocumentStore | None = None,
) -> AttemptLifecycleStore:
    """Platform composition boundary: storage capability → attempt lifecycle store."""
    if kv_store is not None and document_store is not None:
        raise ValueError(
            "wire_attempt_lifecycle_store accepts kv_store or document_store, not both",
        )
    if kv_store is not None:
        return KvAttemptLifecycleStore(kv_store)
    if document_store is not None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "attempt lifecycle persistence requires ConditionalDocumentStore",
            )
        return DocumentStoreAttemptLifecycleStore(document_store)
    raise ValueError(
        "wire_attempt_lifecycle_store requires kv_store or document_store",
    )

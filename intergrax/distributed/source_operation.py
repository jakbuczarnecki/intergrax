# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral source-operation coordination with fencing-safe leases."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)

_SCHEMA = "source_operation_lease.v1"
_PARTITION_PREFIX = _SCHEMA
_MAX_ACQUIRE_ATTEMPTS = 4


def _require_non_empty(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class RagSourceOperationKey:
    """Exact source ownership boundary used by canonical RAG replacement."""

    tenant_id: str
    namespace: str | None
    workspace_id: str | None
    source_id: str

    def __post_init__(self) -> None:
        _require_non_empty(self.tenant_id, field_name="tenant_id")
        if self.namespace is not None:
            _require_non_empty(self.namespace, field_name="namespace")
        if self.workspace_id is not None:
            _require_non_empty(self.workspace_id, field_name="workspace_id")
        _require_non_empty(self.source_id, field_name="source_id")

    @property
    def canonical_value(self) -> str:
        return json.dumps(
            {
                "tenant_id": self.tenant_id,
                "namespace": self.namespace,
                "workspace_id": self.workspace_id,
                "source_id": self.source_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def storage_id(self) -> str:
        return hashlib.sha256(self.canonical_value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceOperationLease:
    key: RagSourceOperationKey
    owner_id: str
    token: str

    def __post_init__(self) -> None:
        _require_non_empty(self.owner_id, field_name="owner_id")
        _require_non_empty(self.token, field_name="token")


@runtime_checkable
class SourceOperationCoordinator(Protocol):
    """Acquire, fence and release one exact source-operation ownership key."""

    def acquire(self, *, key: RagSourceOperationKey) -> SourceOperationLease | None:
        """Return a lease, or ``None`` for a stable conflict."""
        ...

    def release(self, *, lease: SourceOperationLease) -> None:
        """Release only the exact owner/token that was acquired."""
        ...

    def is_owned(self, *, lease: SourceOperationLease) -> bool:
        """Return whether the owner/token can still publish or clean up."""
        ...


class InProcessSourceOperationCoordinator:
    """Thread-safe local coordinator for offline and single-process use."""

    def __init__(
        self,
        *,
        owner_id: str = "in-process",
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        self._owner_id = _require_non_empty(owner_id, field_name="owner_id")
        self._token_factory = token_factory or (lambda: secrets.token_urlsafe(24))
        self._lock = threading.Lock()
        self._leases: dict[RagSourceOperationKey, SourceOperationLease] = {}

    def acquire(self, *, key: RagSourceOperationKey) -> SourceOperationLease | None:
        with self._lock:
            if key in self._leases:
                return None
            lease = SourceOperationLease(
                key=key,
                owner_id=self._owner_id,
                token=_require_non_empty(
                    str(self._token_factory()),
                    field_name="token",
                ),
            )
            self._leases[key] = lease
            return lease

    def release(self, *, lease: SourceOperationLease) -> None:
        with self._lock:
            if self._leases.get(lease.key) == lease:
                del self._leases[lease.key]

    def is_owned(self, *, lease: SourceOperationLease) -> bool:
        with self._lock:
            return self._leases.get(lease.key) == lease


class DocumentStoreSourceOperationCoordinator:
    """Durable CAS-backed coordinator for processes and workers sharing a store."""

    def __init__(
        self,
        document_store: DocumentStore,
        *,
        owner_id: str,
        ttl_seconds: int = 300,
        clock: Callable[[], float] | None = None,
        token_factory: Callable[[], str] | None = None,
        version_factory: Callable[[], str] | None = None,
    ) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError("source operation coordinator requires ConditionalDocumentStore")
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be >= 1")
        self._store = document_store
        self._owner_id = _require_non_empty(owner_id, field_name="owner_id")
        self._ttl_seconds = ttl_seconds
        self._clock = clock or time.time
        self._token_factory = token_factory or (lambda: secrets.token_urlsafe(24))
        self._version_factory = version_factory or (lambda: secrets.token_urlsafe(12))

    def acquire(self, *, key: RagSourceOperationKey) -> SourceOperationLease | None:
        partition_key, row_key = self._location(key)
        for _ in range(_MAX_ACQUIRE_ATTEMPTS):
            now = float(self._clock())
            token = _require_non_empty(str(self._token_factory()), field_name="token")
            candidate = self._document(
                key=key,
                owner_id=self._owner_id,
                token=token,
                expires_at=now + self._ttl_seconds,
                record_version=_require_non_empty(
                    str(self._version_factory()),
                    field_name="record_version",
                ),
            )
            if self._store.put_if_absent(candidate):
                return SourceOperationLease(key, self._owner_id, token)
            existing = self._store.get(partition_key, row_key)
            if existing is None:
                continue
            parsed = self._parse(existing, key=key)
            if float(self._clock()) < parsed["expires_at"]:
                return None
            if self._store.replace_if_match(expected=existing, replacement=candidate):
                return SourceOperationLease(key, self._owner_id, token)
        return None

    def release(self, *, lease: SourceOperationLease) -> None:
        existing = self._store.get(*self._location(lease.key))
        if existing is None:
            return
        parsed = self._parse(existing, key=lease.key)
        if parsed["owner_id"] != lease.owner_id or parsed["token"] != lease.token:
            return
        self._store.delete_if_match(expected=existing)

    def is_owned(self, *, lease: SourceOperationLease) -> bool:
        existing = self._store.get(*self._location(lease.key))
        if existing is None:
            return False
        parsed = self._parse(existing, key=lease.key)
        return (
            parsed["owner_id"] == lease.owner_id
            and parsed["token"] == lease.token
            and float(self._clock()) < parsed["expires_at"]
        )

    @staticmethod
    def _location(key: RagSourceOperationKey) -> tuple[str, str]:
        return (
            f"{_PARTITION_PREFIX}:{key.tenant_id}",
            f"operation:{key.storage_id}",
        )

    def _document(
        self,
        *,
        key: RagSourceOperationKey,
        owner_id: str,
        token: str,
        expires_at: float,
        record_version: str,
    ) -> DocumentRecord:
        return DocumentRecord(
            partition_key=self._location(key)[0],
            row_key=self._location(key)[1],
            data={
                "schema_version": _SCHEMA,
                "key": key.canonical_value,
                "tenant_id": key.tenant_id,
                "owner_id": owner_id,
                "token": token,
                "expires_at": expires_at,
                "record_version": record_version,
            },
            ttl_seconds=self._ttl_seconds,
        )

    @staticmethod
    def _parse(
        record: DocumentRecord,
        *,
        key: RagSourceOperationKey,
    ) -> dict[str, object]:
        if (
            record.data.get("schema_version") != _SCHEMA
            or record.data.get("key") != key.canonical_value
            or record.partition_key != f"{_PARTITION_PREFIX}:{key.tenant_id}"
            or record.row_key != f"operation:{key.storage_id}"
        ):
            raise ValueError("source operation lease state is corrupt")
        owner_id = record.data.get("owner_id")
        token = record.data.get("token")
        expires_at = record.data.get("expires_at")
        if (
            not isinstance(owner_id, str)
            or not isinstance(token, str)
            or not isinstance(expires_at, (int, float))
            or isinstance(expires_at, bool)
        ):
            raise ValueError("source operation lease state is corrupt")
        return {
            "owner_id": owner_id,
            "token": token,
            "expires_at": float(expires_at),
        }

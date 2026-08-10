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
_PUBLICATION_SCHEMA = "source_operation_publication.v1"
_PUBLICATION_PARTITION_PREFIX = _PUBLICATION_SCHEMA
_MAX_ACQUIRE_ATTEMPTS = 4
SOURCE_PUBLICATION_GENERATION_METADATA_KEY = (
    "__intergrax_source_publication_generation"
)


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
    generation: int = 1

    def __post_init__(self) -> None:
        _require_non_empty(self.owner_id, field_name="owner_id")
        _require_non_empty(self.token, field_name="token")
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("generation must be a positive exact int")

    @property
    def publication_generation(self) -> str:
        return f"{self.generation}:{self.token}"


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

    def publication_generation(self, *, lease: SourceOperationLease) -> str:
        """Return the generation identity for writes under this lease."""
        ...

    def active_publication_generation(self, *, key: RagSourceOperationKey) -> str | None:
        """Return the currently visible publication generation for a source."""
        ...

    def promote_publication(self, *, lease: SourceOperationLease) -> bool:
        """CAS-promote this generation to the active publication."""
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
        self._generations: dict[RagSourceOperationKey, int] = {}
        self._active_publications: dict[RagSourceOperationKey, SourceOperationLease] = {}

    def acquire(self, *, key: RagSourceOperationKey) -> SourceOperationLease | None:
        with self._lock:
            if key in self._leases:
                return None
            generation = self._generations.get(key, 0) + 1
            self._generations[key] = generation
            lease = SourceOperationLease(
                key=key,
                owner_id=self._owner_id,
                token=_require_non_empty(
                    str(self._token_factory()),
                    field_name="token",
                ),
                generation=generation,
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

    def publication_generation(self, *, lease: SourceOperationLease) -> str:
        with self._lock:
            if self._leases.get(lease.key) != lease:
                raise RuntimeError("source operation lease is no longer owned")
            return lease.publication_generation

    def active_publication_generation(
        self,
        *,
        key: RagSourceOperationKey,
    ) -> str | None:
        with self._lock:
            active = self._active_publications.get(key)
            return active.publication_generation if active is not None else None

    def promote_publication(self, *, lease: SourceOperationLease) -> bool:
        with self._lock:
            if self._leases.get(lease.key) != lease:
                return False
            active = self._active_publications.get(lease.key)
            if active is not None and active.generation >= lease.generation:
                return False
            self._active_publications[lease.key] = lease
            return True


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
            existing = self._store.get(partition_key, row_key)
            active = self._read_publication(key)
            token = _require_non_empty(str(self._token_factory()), field_name="token")
            if existing is None:
                generation = (int(active["generation"]) if active is not None else 0) + 1
                candidate = self._document(
                    key=key,
                    owner_id=self._owner_id,
                    token=token,
                    generation=generation,
                    expires_at=now + self._ttl_seconds,
                    record_version=_require_non_empty(
                        str(self._version_factory()),
                        field_name="record_version",
                    ),
                )
                if self._store.put_if_absent(candidate):
                    return SourceOperationLease(
                        key,
                        self._owner_id,
                        token,
                        generation,
                    )
                continue
            parsed = self._parse(existing, key=key)
            if float(self._clock()) < parsed["expires_at"]:
                return None
            generation = max(
                int(parsed["generation"]),
                int(active["generation"]) if active is not None else 0,
            ) + 1
            candidate = self._document(
                key=key,
                owner_id=self._owner_id,
                token=token,
                generation=generation,
                expires_at=now + self._ttl_seconds,
                record_version=_require_non_empty(
                    str(self._version_factory()),
                    field_name="record_version",
                ),
            )
            if self._store.replace_if_match(expected=existing, replacement=candidate):
                return SourceOperationLease(
                    key,
                    self._owner_id,
                    token,
                    generation,
                )
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
            and parsed["generation"] == lease.generation
            and float(self._clock()) < parsed["expires_at"]
        )

    def publication_generation(self, *, lease: SourceOperationLease) -> str:
        if not self.is_owned(lease=lease):
            raise RuntimeError("source operation lease is no longer owned")
        return lease.publication_generation

    def active_publication_generation(
        self,
        *,
        key: RagSourceOperationKey,
    ) -> str | None:
        publication = self._read_publication(key)
        if publication is None:
            return None
        return _publication_identity(
            int(publication["generation"]),
            str(publication["token"]),
        )

    def promote_publication(self, *, lease: SourceOperationLease) -> bool:
        if not self.is_owned(lease=lease):
            return False
        for _ in range(_MAX_ACQUIRE_ATTEMPTS):
            if not self.is_owned(lease=lease):
                return False
            existing = self._read_publication_record(lease.key)
            if existing is not None:
                parsed = self._parse_publication(existing, key=lease.key)
                if int(parsed["generation"]) >= lease.generation:
                    return False
            candidate = self._publication_document(lease=lease)
            if existing is None:
                if self._store.put_if_absent(candidate):
                    return True
            elif self._store.replace_if_match(
                expected=existing,
                replacement=candidate,
            ):
                return True
        return False

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
        generation: int,
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
                "generation": generation,
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
        generation = record.data.get("generation", 1)
        expires_at = record.data.get("expires_at")
        if (
            not isinstance(owner_id, str)
            or not isinstance(token, str)
            or type(generation) is not int
            or generation < 1
            or not isinstance(expires_at, (int, float))
            or isinstance(expires_at, bool)
        ):
            raise ValueError("source operation lease state is corrupt")
        return {
            "owner_id": owner_id,
            "token": token,
            "generation": generation,
            "expires_at": float(expires_at),
        }

    @staticmethod
    def _publication_location(key: RagSourceOperationKey) -> tuple[str, str]:
        return (
            f"{_PUBLICATION_PARTITION_PREFIX}:{key.tenant_id}",
            f"publication:{key.storage_id}",
        )

    def _read_publication_record(
        self,
        key: RagSourceOperationKey,
    ) -> DocumentRecord | None:
        return self._store.get(*self._publication_location(key))

    def _read_publication(
        self,
        key: RagSourceOperationKey,
    ) -> dict[str, object] | None:
        record = self._read_publication_record(key)
        if record is None:
            return None
        return self._parse_publication(record, key=key)

    def _publication_document(self, *, lease: SourceOperationLease) -> DocumentRecord:
        partition_key, row_key = self._publication_location(lease.key)
        return DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data={
                "schema_version": _PUBLICATION_SCHEMA,
                "key": lease.key.canonical_value,
                "generation": lease.generation,
                "token": lease.token,
            },
            ttl_seconds=None,
        )

    @staticmethod
    def _parse_publication(
        record: DocumentRecord,
        *,
        key: RagSourceOperationKey,
    ) -> dict[str, object]:
        if (
            record.data.get("schema_version") != _PUBLICATION_SCHEMA
            or record.data.get("key") != key.canonical_value
            or record.partition_key != f"{_PUBLICATION_PARTITION_PREFIX}:{key.tenant_id}"
            or record.row_key != f"publication:{key.storage_id}"
        ):
            raise ValueError("source publication state is corrupt")
        generation = record.data.get("generation")
        token = record.data.get("token")
        if type(generation) is not int or generation < 1 or not isinstance(token, str):
            raise ValueError("source publication state is corrupt")
        return {"generation": generation, "token": token}


def _publication_identity(generation: int, token: str) -> str:
    return f"{generation}:{token}"

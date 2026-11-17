# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class KVDurabilityMechanism(StrEnum):
    """Closed set of persistence mechanisms an operator may attest for a KV backend."""

    REDIS_AOF_FSYNC_ALWAYS = "redis_aof_fsync_always"
    REDIS_AOF_FSYNC_EVERYSEC = "redis_aof_fsync_everysec"


@dataclass(frozen=True, slots=True)
class KVDurabilityAttestation:
    """
    Typed, explicit, operator-stated claim that a KV backend persists across process restart.

    The claim is declared, not probed: construction checks only the shape of the fields.
    """

    mechanism: KVDurabilityMechanism
    attested_by: str
    procedure_ref: str

    def __post_init__(self) -> None:
        if not isinstance(self.mechanism, KVDurabilityMechanism):
            raise ValueError("KVDurabilityAttestation.mechanism must be a KVDurabilityMechanism")
        if not isinstance(self.attested_by, str) or not self.attested_by.strip():
            raise ValueError("KVDurabilityAttestation.attested_by must be a non-empty string")
        if not isinstance(self.procedure_ref, str) or not self.procedure_ref.strip():
            raise ValueError("KVDurabilityAttestation.procedure_ref must be a non-empty string")


class DistributedKVStore(ABC):
    """
    Abstract distributed key-value store contract.

    Guarantees:
    - tenant isolation
    - atomic write primitives
    - crash-safe semantics (backend dependent)

    This contract is backend-agnostic (Redis, Dragonfly, etc.).
    """

    @property
    def survives_process_restart(self) -> bool:
        """
        Whether data written through this store survives process restart.

        Default ``False``: a backend must opt in explicitly (fail-closed).
        """
        return False

    @abstractmethod
    def get(
        self,
        tenant_id: str,
        key: str,
    ) -> bytes | None:
        """
        Retrieve value for given tenant and key.
        Returns None if key does not exist.
        """
        ...

    @abstractmethod
    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        """
        Set value for given tenant and key.
        Optional TTL in seconds.
        """
        ...

    @abstractmethod
    def delete(
        self,
        tenant_id: str,
        key: str,
    ) -> None:
        """
        Delete key for given tenant.
        """
        ...

    @abstractmethod
    def compare_and_set(
        self,
        tenant_id: str,
        key: str,
        expected: bytes | None,
        new_value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        """
        Atomic compare-and-set operation.

        Returns True if update succeeded, False otherwise.
        """
        ...


def kv_store_survives_process_restart(store: object) -> bool:
    """True only when ``store.survives_process_restart`` is exactly ``True`` (no truthy coercion)."""
    return getattr(store, "survives_process_restart", False) is True


@runtime_checkable
class DistributedKVStoreProvider(Protocol):
    """
    Explicit platform contract for integrations that expose a ``DistributedKVStore``.

    Queue host composition resolves KV via ``isinstance`` against this protocol
    (structural, runtime-checkable) — not dynamic attribute lookup.
    """

    @property
    def kv_store(self) -> DistributedKVStore: ...

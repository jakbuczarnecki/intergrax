# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Protocol, runtime_checkable


class DistributedKVStore(ABC):
    """
    Abstract distributed key-value store contract.

    Guarantees:
    - tenant isolation
    - atomic write primitives
    - crash-safe semantics (backend dependent)

    This contract is backend-agnostic (Redis, Dragonfly, etc.).
    """

    @abstractmethod
    def get(
        self,
        tenant_id: str,
        key: str,
    ) -> Optional[bytes]:
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
        ttl_seconds: Optional[int] = None,
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
        expected: Optional[bytes],
        new_value: bytes,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """
        Atomic compare-and-set operation.

        Returns True if update succeeded, False otherwise.
        """
        ...


@runtime_checkable
class DistributedKVStoreProvider(Protocol):
    """
    Explicit platform contract for integrations that expose a ``DistributedKVStore``.

    Queue host composition resolves KV via ``isinstance`` against this protocol
    (structural, runtime-checkable) — not dynamic attribute lookup.
    """

    @property
    def kv_store(self) -> DistributedKVStore: ...
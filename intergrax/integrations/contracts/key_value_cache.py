# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Key-value cache integration contract (§7.1.2, Phase M.2)."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class KeyValueCache(Protocol):
    """
    Tenant-scoped cache / lock / idempotency primitive.

    Aligns with ``intergrax.distributed.contracts.kv_store.DistributedKVStore``;
    providers may wrap that ABC via ``set_if_absent`` ↔ ``compare_and_set``.
    """

    def get(self, tenant_id: str, key: str) -> Optional[bytes]:
        """Return value or None when missing."""

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Store value with optional TTL."""

    def delete(self, tenant_id: str, key: str) -> None:
        """Remove key for tenant."""

    def set_if_absent(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Return True when the key was created, False when it already existed."""

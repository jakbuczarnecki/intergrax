# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.integrations.contracts.key_value_cache import KeyValueCache


class InProcessKeyValueCache(KeyValueCache):
    """Minimal KeyValueCache for documentation and tests."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}

    def get(self, tenant_id: str, key: str) -> bytes | None:
        return self._data.get((tenant_id, key))

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        _ = ttl_seconds
        self._data[(tenant_id, key)] = value

    def delete(self, tenant_id: str, key: str) -> None:
        self._data.pop((tenant_id, key), None)

    def set_if_absent(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        ttl_seconds: int | None = None,
    ) -> bool:
        if (tenant_id, key) in self._data:
            return False
        self.set(tenant_id, key, value, ttl_seconds=ttl_seconds)
        return True

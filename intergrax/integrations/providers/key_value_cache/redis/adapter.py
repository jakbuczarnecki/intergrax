# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Redis key-value cache adapter — wraps ``distributed.providers.redis_kv_store``."""

from __future__ import annotations

from typing import Optional

from intergrax.distributed.providers.redis_kv_store import RedisKVStore
from intergrax.integrations.contracts.key_value_cache import KeyValueCache


class RedisKeyValueCache:
    """
    Integration-catalog facade over ``RedisKVStore``.

    Implements ``KeyValueCache``; exposes ``kv_store`` for queueing / transport wiring.
    Instantiate via ``create_redis_integration()`` or ``create_redis_key_value_cache()``
    in ``bundle.py`` — not directly.
    """

    def __init__(self, store: RedisKVStore) -> None:
        self._store = store

    @property
    def kv_store(self) -> RedisKVStore:
        return self._store

    def get(self, tenant_id: str, key: str) -> Optional[bytes]:
        return self._store.get(tenant_id, key)

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        self._store.set(tenant_id, key, value, ttl_seconds=ttl_seconds)

    def delete(self, tenant_id: str, key: str) -> None:
        self._store.delete(tenant_id, key)

    def set_if_absent(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        return self._store.compare_and_set(
            tenant_id,
            key,
            expected=None,
            new_value=value,
            ttl_seconds=ttl_seconds,
        )

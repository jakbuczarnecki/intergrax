# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

import redis

from intergrax.distributed.contracts.kv_store import DistributedKVStore


class RedisKVStore(DistributedKVStore):
    """
    Redis-based implementation of DistributedKVStore.

    This implementation is synchronous and uses redis-py client.
    Tenant isolation is enforced via key prefixing.
    """

    def __init__(
        self,
        client: redis.Redis,
        *,
        key_prefix: str = "intergrax",
    ) -> None:
        self._client = client
        self._key_prefix = key_prefix

    def _build_key(
        self,
        tenant_id: str,
        key: str,
    ) -> str:
        return f"{self._key_prefix}:{tenant_id}:{key}"

    def get(
        self,
        tenant_id: str,
        key: str,
    ) -> Optional[bytes]:
        redis_key = self._build_key(tenant_id, key)
        return self._client.get(redis_key)

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        redis_key = self._build_key(tenant_id, key)

        if ttl_seconds is not None:
            self._client.set(redis_key, value, ex=ttl_seconds)
        else:
            self._client.set(redis_key, value)

    def delete(
        self,
        tenant_id: str,
        key: str,
    ) -> None:
        redis_key = self._build_key(tenant_id, key)
        self._client.delete(redis_key)

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
        Minimal implementation placeholder.
        Atomic semantics will be implemented in a dedicated step.
        """
        raise NotImplementedError("compare_and_set not yet implemented.")
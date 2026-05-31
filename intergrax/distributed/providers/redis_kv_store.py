# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Redis-based implementation of DistributedKVStore.

Composition root: ``intergrax.integrations.providers.key_value_cache.redis.create_redis_integration``.
Do not instantiate directly from application code.
"""

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
        Atomic compare-and-set using Redis Lua script.

        Semantics:
        - If expected is None: succeeds only if key does not exist.
        - If expected is not None: succeeds only if current value equals expected.
        - On success: sets new_value (with optional TTL).
        - Returns True on success, False otherwise.
        """

        redis_key = self._build_key(tenant_id, key)

        lua_script = """
        local current = redis.call("GET", KEYS[1])

        if ARGV[1] == "__NONE__" then
            if current == false then
                if ARGV[3] ~= "" then
                    redis.call("SET", KEYS[1], ARGV[2], "EX", ARGV[3])
                else
                    redis.call("SET", KEYS[1], ARGV[2])
                end
                return 1
            else
                return 0
            end
        else
            if current == ARGV[1] then
                if ARGV[3] ~= "" then
                    redis.call("SET", KEYS[1], ARGV[2], "EX", ARGV[3])
                else
                    redis.call("SET", KEYS[1], ARGV[2])
                end
                return 1
            else
                return 0
            end
        end
        """

        expected_arg: bytes
        if expected is None:
            expected_arg = b"__NONE__"
        else:
            expected_arg = expected

        ttl_arg = ""
        if ttl_seconds is not None:
            ttl_arg = str(ttl_seconds)

        result = self._client.eval(
            lua_script,
            1,
            redis_key,
            expected_arg,
            new_value,
            ttl_arg,
        )

        return result == 1
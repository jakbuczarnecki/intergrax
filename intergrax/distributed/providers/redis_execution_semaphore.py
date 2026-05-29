# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Redis-backed distributed execution semaphore.

Composition root: ``intergrax.integrations.providers.redis.create_redis_execution_semaphore``.
"""

from __future__ import annotations

import uuid
from typing import Optional

import redis

from intergrax.distributed.contracts.execution_semaphore import (
    DistributedExecutionSemaphore,
    ExecutionSlot,
)


class RedisExecutionSemaphore(DistributedExecutionSemaphore):
    """
    Redis-backed distributed execution semaphore.

    Uses Redis SET to track active execution slots.
    Atomicity guaranteed via Lua script.
    """

    def __init__(
        self,
        *,
        client: redis.Redis,
        ttl_seconds: int = 300,
    ) -> None:
        self._client = client
        self._ttl_seconds = ttl_seconds

        self._acquire_script = self._client.register_script(
            """
            local key = KEYS[1]
            local max_parallel = tonumber(ARGV[1])
            local slot_id = ARGV[2]
            local ttl = tonumber(ARGV[3])

            local current = redis.call("SCARD", key)

            if current < max_parallel then
                redis.call("SADD", key, slot_id)
                redis.call("EXPIRE", key, ttl)
                return 1
            else
                return 0
            end
            """
        )

    def acquire(
        self,
        *,
        tenant_id: str,
        max_parallel: int,
    ) -> Optional[ExecutionSlot]:
        key = f"execution:{tenant_id}"
        slot_id = str(uuid.uuid4())

        granted = self._acquire_script(
            keys=[key],
            args=[max_parallel, slot_id, self._ttl_seconds],
        )

        if int(granted) == 1:
            return ExecutionSlot(slot_id=slot_id)

        return None

    def release(
        self,
        *,
        tenant_id: str,
        slot: ExecutionSlot,
    ) -> None:
        key = f"execution:{tenant_id}"

        pipe = self._client.pipeline()
        pipe.srem(key, slot.slot_id)
        pipe.scard(key)
        removed, remaining = pipe.execute()

        if int(remaining) == 0:
            self._client.delete(key)
# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations


from redis import Redis

from intergrax.distributed.contracts.rate_limiter import (
    DistributedRateLimiter,
    RateLimitResult,
)


class RedisDistributedRateLimiter(DistributedRateLimiter):
    """
    Redis-backed token bucket rate limiter.

    Uses atomic Lua script for correctness under concurrency.
    """

    _LUA_SCRIPT = """
    local key = KEYS[1]
    local capacity = tonumber(ARGV[1])
    local refill_rate = tonumber(ARGV[2])

    -- Get current Redis time
    local now_data = redis.call("TIME")
    local now = tonumber(now_data[1]) + tonumber(now_data[2]) / 1000000

    local bucket = redis.call("HMGET", key, "tokens", "last_refill")
    local tokens = tonumber(bucket[1])
    local last_refill = tonumber(bucket[2])

    if tokens == nil then
        tokens = capacity
        last_refill = now
    end

    -- Refill tokens
    local elapsed = now - last_refill
    local refill = elapsed * refill_rate
    tokens = math.min(capacity, tokens + refill)

    local allowed = 0
    local retry_after = 0

    if tokens >= 1 then
        tokens = tokens - 1
        allowed = 1
    else
        local missing = 1 - tokens
        retry_after = missing / refill_rate
    end

    -- Update bucket
    redis.call("HMSET", key,
        "tokens", tokens,
        "last_refill", now
    )

    -- Set TTL (optional cleanup)
    local ttl

    if refill_rate > 0 then
        ttl = math.ceil(capacity / refill_rate * 2)
    else
        -- No refill: use fixed TTL to avoid division by zero
        ttl = 60
    end

    redis.call("EXPIRE", key, ttl)

    return {allowed, tokens, retry_after}
    """

    def __init__(self, redis_client: Redis) -> None:
        self._redis: Redis = redis_client
        self._script = self._redis.register_script(self._LUA_SCRIPT)

    def acquire(
        self,
        *,
        tenant_id: str,
        key: str,
        capacity: int,
        refill_rate_per_second: float,
    ) -> RateLimitResult:

        redis_key = f"rate:{tenant_id}:{key}"

        result = self._script(
            keys=[redis_key],
            args=[capacity, refill_rate_per_second],
        )

        allowed = bool(result[0])
        remaining_tokens = float(result[1])
        retry_after_seconds = float(result[2])

        return RateLimitResult(
            allowed=allowed,
            remaining_tokens=remaining_tokens,
            retry_after_seconds=retry_after_seconds,
        )
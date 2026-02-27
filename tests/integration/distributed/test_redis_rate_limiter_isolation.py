# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import uuid

import concurrent.futures
import pytest
from redis import Redis

from intergrax.distributed.providers.redis_rate_limiter import (
    RedisDistributedRateLimiter,
)

pytestmark = pytest.mark.integration


def test_redis_rate_limiter_multi_tenant_isolation() -> None:
    """
    Verifies that token buckets are isolated per tenant.
    """

    redis = Redis(host="localhost", port=6379, decode_responses=False)

    limiter = RedisDistributedRateLimiter(redis_client=redis)

    # Unique key to avoid collisions across test runs
    logical_key = f"test_rl:{uuid.uuid4()}"

    capacity = 1
    refill_rate = 0.0  # No refill

    tenant_a = "tenant_A"
    tenant_b = "tenant_B"

    # Tenant A - first acquire should pass
    result_a1 = limiter.acquire(
        tenant_id=tenant_a,
        key=logical_key,
        capacity=capacity,
        refill_rate_per_second=refill_rate,
    )
    assert result_a1.allowed is True

    # Tenant A - second acquire should fail
    result_a2 = limiter.acquire(
        tenant_id=tenant_a,
        key=logical_key,
        capacity=capacity,
        refill_rate_per_second=refill_rate,
    )
    assert result_a2.allowed is False

    # Tenant B - first acquire should still pass (isolation)
    result_b1 = limiter.acquire(
        tenant_id=tenant_b,
        key=logical_key,
        capacity=capacity,
        refill_rate_per_second=refill_rate,
    )
    assert result_b1.allowed is True

    # Cleanup (best-effort)
    redis.delete(f"rate:{tenant_a}:{logical_key}")
    redis.delete(f"rate:{tenant_b}:{logical_key}")



def test_redis_rate_limiter_atomic_concurrency() -> None:
    """
    Verifies that concurrent acquire calls are atomic.
    Only one call should be allowed when capacity=1 and no refill.
    """

    redis = Redis(host="localhost", port=6379, decode_responses=False)

    limiter = RedisDistributedRateLimiter(redis_client=redis)

    logical_key = f"test_rl_atomic:{uuid.uuid4()}"

    capacity = 1
    refill_rate = 0.0

    tenant = "tenant_atomic"

    def attempt_acquire() -> bool:
        result = limiter.acquire(
            tenant_id=tenant,
            key=logical_key,
            capacity=capacity,
            refill_rate_per_second=refill_rate,
        )
        return result.allowed

    # Run 10 concurrent attempts
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda _: attempt_acquire(), range(10)))

    allowed_count = sum(1 for r in results if r is True)

    assert allowed_count == 1

    # Cleanup
    redis.delete(f"rate:{tenant}:{logical_key}")
# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import time

import redis
import pytest

from intergrax.distributed.providers.redis_rate_limiter import RedisDistributedRateLimiter

pytestmark = pytest.mark.integration


@pytest.fixture()
def redis_client() -> redis.Redis:
    client = redis.Redis(host="localhost", port=6379, db=15)
    client.flushdb()
    return client


@pytest.fixture()
def rate_limiter(redis_client: redis.Redis) -> RedisDistributedRateLimiter:
    return RedisDistributedRateLimiter(redis_client=redis_client)


def test_acquire_allows_initial_request(rate_limiter: RedisDistributedRateLimiter) -> None:
    result = rate_limiter.acquire(
        tenant_id="tenant",
        key="llm",
        capacity=2,
        refill_rate_per_second=1.0,
    )

    assert result.allowed is True
    assert result.remaining_tokens >= 0.0
    assert result.retry_after_seconds == 0.0


def test_acquire_denies_when_bucket_exhausted(rate_limiter: RedisDistributedRateLimiter) -> None:
    capacity = 2

    for _ in range(capacity):
        r = rate_limiter.acquire(
            tenant_id="tenant",
            key="llm",
            capacity=capacity,
            refill_rate_per_second=0.0001,  # refill irrelevant within same tick
        )
        assert r.allowed is True

    denied = rate_limiter.acquire(
        tenant_id="tenant",
        key="llm",
        capacity=capacity,
        refill_rate_per_second=0.0001,
    )

    assert denied.allowed is False
    assert denied.retry_after_seconds > 0.0


def test_acquire_refills_over_time(rate_limiter: RedisDistributedRateLimiter) -> None:
    capacity = 1
    refill_rate = 5.0  # 1 token in 0.2s

    first = rate_limiter.acquire(
        tenant_id="tenant",
        key="websearch",
        capacity=capacity,
        refill_rate_per_second=refill_rate,
    )
    assert first.allowed is True

    denied = rate_limiter.acquire(
        tenant_id="tenant",
        key="websearch",
        capacity=capacity,
        refill_rate_per_second=refill_rate,
    )
    assert denied.allowed is False

    # wait enough for one token to refill
    time.sleep(0.25)

    allowed_again = rate_limiter.acquire(
        tenant_id="tenant",
        key="websearch",
        capacity=capacity,
        refill_rate_per_second=refill_rate,
    )
    assert allowed_again.allowed is True


def test_acquire_isolated_per_tenant(rate_limiter: RedisDistributedRateLimiter) -> None:
    capacity = 1
    refill_rate = 0.0001  # effectively no refill within test time

    t1_first = rate_limiter.acquire(
        tenant_id="tenant_A",
        key="llm",
        capacity=capacity,
        refill_rate_per_second=refill_rate,
    )
    assert t1_first.allowed is True

    t1_denied = rate_limiter.acquire(
        tenant_id="tenant_A",
        key="llm",
        capacity=capacity,
        refill_rate_per_second=refill_rate,
    )
    assert t1_denied.allowed is False

    # Another tenant should still be allowed for the same logical key
    t2_first = rate_limiter.acquire(
        tenant_id="tenant_B",
        key="llm",
        capacity=capacity,
        refill_rate_per_second=refill_rate,
    )
    assert t2_first.allowed is True
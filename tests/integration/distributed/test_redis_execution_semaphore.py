# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Generator

from redis import Redis
import pytest

from intergrax.distributed.providers.redis_execution_semaphore import (
    RedisExecutionSemaphore,
)

pytestmark = pytest.mark.integration

REDIS_URL = "redis://localhost:6379/0"


@pytest.fixture
def redis_client() -> Generator[Redis, None, None]:
    client = Redis.from_url(REDIS_URL)
    client.flushdb()
    yield client
    client.flushdb()


def test_acquire_respects_max_parallel(redis_client: Redis) -> None:
    semaphore = RedisExecutionSemaphore(
        client=redis_client,
        ttl_seconds=60,
    )

    slot1 = semaphore.acquire(tenant_id="tenant_A", max_parallel=2)
    slot2 = semaphore.acquire(tenant_id="tenant_A", max_parallel=2)
    slot3 = semaphore.acquire(tenant_id="tenant_A", max_parallel=2)

    assert slot1 is not None
    assert slot2 is not None
    assert slot3 is None  # limit reached


def test_release_frees_slot(redis_client: Redis) -> None:
    semaphore = RedisExecutionSemaphore(
        client=redis_client,
        ttl_seconds=60,
    )

    slot1 = semaphore.acquire(tenant_id="tenant_B", max_parallel=1)
    assert slot1 is not None

    slot2 = semaphore.acquire(tenant_id="tenant_B", max_parallel=1)
    assert slot2 is None  # blocked

    semaphore.release(tenant_id="tenant_B", slot=slot1)

    slot3 = semaphore.acquire(tenant_id="tenant_B", max_parallel=1)
    assert slot3 is not None  # slot freed


def test_release_is_idempotent(redis_client: Redis) -> None:
    semaphore = RedisExecutionSemaphore(
        client=redis_client,
        ttl_seconds=60,
    )

    slot = semaphore.acquire(tenant_id="tenant_C", max_parallel=1)
    assert slot is not None

    semaphore.release(tenant_id="tenant_C", slot=slot)
    semaphore.release(tenant_id="tenant_C", slot=slot)  # second release

    slot2 = semaphore.acquire(tenant_id="tenant_C", max_parallel=1)
    assert slot2 is not None
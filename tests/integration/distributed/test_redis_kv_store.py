# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import uuid

import redis
import pytest

from intergrax.distributed.providers.redis_kv_store import RedisKVStore

pytestmark = pytest.mark.integration

@pytest.fixture()
def redis_client() -> redis.Redis:
    client = redis.Redis(host="localhost", port=6379, db=15)
    client.flushdb()
    return client


@pytest.fixture()
def kv_store(redis_client: redis.Redis) -> RedisKVStore:
    return RedisKVStore(client=redis_client, key_prefix="test")


def _unique_key() -> str:
    return f"key:{uuid.uuid4().hex}"


def test_compare_and_set_when_key_does_not_exist_and_expected_none(
    kv_store: RedisKVStore,
) -> None:
    key = _unique_key()

    success = kv_store.compare_and_set(
        tenant_id="tenant",
        key=key,
        expected=None,
        new_value=b"value1",
    )

    assert success is True
    assert kv_store.get("tenant", key) == b"value1"


def test_compare_and_set_fails_when_key_exists_and_expected_none(
    kv_store: RedisKVStore,
) -> None:
    key = _unique_key()

    kv_store.set("tenant", key, b"value1")

    success = kv_store.compare_and_set(
        tenant_id="tenant",
        key=key,
        expected=None,
        new_value=b"value2",
    )

    assert success is False
    assert kv_store.get("tenant", key) == b"value1"


def test_compare_and_set_succeeds_when_expected_matches(
    kv_store: RedisKVStore,
) -> None:
    key = _unique_key()

    kv_store.set("tenant", key, b"value1")

    success = kv_store.compare_and_set(
        tenant_id="tenant",
        key=key,
        expected=b"value1",
        new_value=b"value2",
    )

    assert success is True
    assert kv_store.get("tenant", key) == b"value2"


def test_compare_and_set_fails_when_expected_does_not_match(
    kv_store: RedisKVStore,
) -> None:
    key = _unique_key()

    kv_store.set("tenant", key, b"value1")

    success = kv_store.compare_and_set(
        tenant_id="tenant",
        key=key,
        expected=b"other",
        new_value=b"value2",
    )

    assert success is False
    assert kv_store.get("tenant", key) == b"value1"


def test_compare_and_set_with_ttl(
    kv_store: RedisKVStore,
) -> None:
    key = _unique_key()

    success = kv_store.compare_and_set(
        tenant_id="tenant",
        key=key,
        expected=None,
        new_value=b"value",
        ttl_seconds=1,
    )

    assert success is True
    assert kv_store.get("tenant", key) == b"value"

    # ensure TTL applied
    ttl = kv_store._client.ttl(
        kv_store._build_key("tenant", key)
    )
    assert ttl > 0
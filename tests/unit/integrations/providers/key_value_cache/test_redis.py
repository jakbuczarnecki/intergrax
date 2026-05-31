# © Artur Czarnecki. All rights reserved.

# Intergrax framework – proprietary and confidential.



"""Unit tests for Redis integration provider (Phase M.4)."""



from __future__ import annotations



from unittest.mock import MagicMock



import pytest



from intergrax.integrations._shared.conformance import assert_key_value_cache

from intergrax.integrations.contracts.base import IntegrationCategory

from intergrax.integrations.providers.key_value_cache.redis.adapter import RedisKeyValueCache

from intergrax.integrations.providers.key_value_cache.redis.bundle import (

    RedisIntegrationBundle,

    create_redis_integration,

    create_redis_key_value_cache,

    create_redis_rerank_cache,

)

from intergrax.integrations.providers.key_value_cache.redis.register import register_redis_integration

from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state

from intergrax.integrations.registry.catalog import clear_catalog

from intergrax.integrations.registry.factory import resolve

from intergrax.integrations.registry.profile import IntegrationProfile

from intergrax.integrations.registry.slugs import IntegrationSlug

from intergrax.distributed.providers.redis_idempotency_store import RedisIdempotencyStore

from intergrax.distributed.providers.redis_kv_store import RedisKVStore

from intergrax.distributed.providers.redis_rate_limiter import RedisDistributedRateLimiter

from intergrax.distributed.providers.redis_execution_semaphore import RedisExecutionSemaphore

from intergrax.rag.rerankers.cache.redis_rerank_cache import RedisRerankCache



pytestmark = pytest.mark.unit





@pytest.fixture(autouse=True)

def _clean_catalog() -> None:

    clear_catalog()

    reset_default_integrations_state()

    yield

    clear_catalog()

    reset_default_integrations_state()





def test_redis_adapter_set_if_absent_delegates_to_compare_and_set() -> None:

    store = MagicMock(spec=RedisKVStore)

    store.compare_and_set.return_value = True

    cache = RedisKeyValueCache(store)



    assert cache.set_if_absent("t1", "k", b"v", ttl_seconds=60) is True

    store.compare_and_set.assert_called_once_with(

        "t1",

        "k",

        expected=None,

        new_value=b"v",

        ttl_seconds=60,

    )





def test_create_redis_key_value_cache_uses_injected_client() -> None:

    client = MagicMock()

    cache = create_redis_key_value_cache(client=client, key_prefix="lab")



    assert_key_value_cache(cache)

    assert cache.kv_store._client is client

    assert cache.kv_store._key_prefix == "lab"





def test_create_redis_integration_bundle_shares_one_client() -> None:

    client = MagicMock()

    bundle = create_redis_integration(client=client, key_prefix="prod")



    assert isinstance(bundle, RedisIntegrationBundle)

    assert bundle.client is client

    assert bundle.key_value_cache.kv_store._client is client

    assert isinstance(bundle.idempotency_store, RedisIdempotencyStore)

    assert isinstance(bundle.rate_limiter, RedisDistributedRateLimiter)

    assert isinstance(bundle.execution_semaphore, RedisExecutionSemaphore)

    assert bundle.kv_store is bundle.key_value_cache.kv_store





def test_create_redis_rerank_cache_uses_shared_client_factory() -> None:

    client = MagicMock()

    cache = create_redis_rerank_cache(client=client)



    assert isinstance(cache, RedisRerankCache)

    assert cache._redis is client





def test_register_and_resolve_via_profile() -> None:

    client = MagicMock()

    register_redis_integration()



    profile = IntegrationProfile(key_value_cache=IntegrationSlug.REDIS)

    cache = resolve(

        IntegrationCategory.KEY_VALUE_CACHE,

        profile=profile,

        config={"client": client, "key_prefix": "test"},

    )



    assert_key_value_cache(cache)

    assert cache.kv_store._key_prefix == "test"





def test_register_default_integrations_includes_redis() -> None:

    register_default_integrations()

    profile = IntegrationProfile(key_value_cache=IntegrationSlug.REDIS)



    cache = resolve(

        IntegrationCategory.KEY_VALUE_CACHE,

        profile=profile,

        config={"client": MagicMock()},

    )



    assert isinstance(cache, RedisKeyValueCache)



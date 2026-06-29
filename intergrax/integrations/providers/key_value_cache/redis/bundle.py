# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete Redis integration bundle — the single composition root for Redis in Intergrax.

All runtime wiring (KV cache, idempotency, rate limits, execution semaphores, RAG rerank
cache) MUST obtain clients and facades from this module or ``resolve(..., KEY_VALUE_CACHE)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.distributed.contracts.execution_semaphore import DistributedExecutionSemaphore
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.distributed.contracts.rate_limiter import DistributedRateLimiter
from intergrax.distributed.providers.redis_execution_semaphore import RedisExecutionSemaphore
from intergrax.distributed.providers.redis_idempotency_store import RedisIdempotencyStore
from intergrax.distributed.providers.redis_kv_store import RedisKVStore
from intergrax.distributed.providers.redis_rate_limiter import RedisDistributedRateLimiter
from intergrax.integrations.contracts.key_value_cache import KeyValueCache
from intergrax.integrations.providers.key_value_cache.redis.adapter import _RedisKeyValueCache
from intergrax.integrations.providers.key_value_cache.redis.client import create_redis_client, resolve_redis_config
from intergrax.integrations.providers.key_value_cache.redis.config import RedisIntegrationConfig
from intergrax.rag.rerankers.cache.base_rerank_cache import BaseRerankCache
from intergrax.rag.rerankers.cache.redis_rerank_cache import RedisRerankCache


@dataclass(frozen=True)
class RedisIntegrationBundle:
    """
    All Redis-backed Tier-0 facades sharing one client and config.

    Use ``create_redis_integration()`` — do not construct Redis facades ad hoc elsewhere.
    """

    client: "redis.Redis"
    config: RedisIntegrationConfig
    key_value_cache: RedisKeyValueCacheIntegration
    idempotency_store: RedisIdempotencyStore
    rate_limiter: RedisDistributedRateLimiter
    execution_semaphore: RedisExecutionSemaphore

    @property
    def kv_store(self) -> DistributedKVStore:
        """``DistributedKVStore`` for queueing / transport (legacy contract)."""
        return self.key_value_cache.kv_store


def create_redis_integration(
    *,
    url: Optional[str] = None,
    db: Optional[int] = None,
    key_prefix: Optional[str] = None,
    client: Optional["redis.Redis"] = None,
    execution_semaphore_ttl_seconds: int = 300,
    **config_overrides: object,
) -> RedisIntegrationBundle:
    """
    Single entry point for Redis — shared client and all distributed facades.

    Registered in the integration catalog for ``key_value_cache`` via
    ``create_redis_key_value_cache``; prefer this function when wiring multiple
    Redis-backed components in Tier-3 composition roots.
    """
    overrides: dict[str, object] = dict(config_overrides)
    if url is not None:
        overrides["url"] = url
    if db is not None:
        overrides["db"] = db
    if key_prefix is not None:
        overrides["key_prefix"] = key_prefix

    config = resolve_redis_config(**overrides)
    resolved_client = create_redis_client(client=client, **config.model_dump())

    store = RedisKVStore(client=resolved_client, key_prefix=config.key_prefix)
    cache = _RedisKeyValueCache(store)

    return RedisIntegrationBundle(
        client=resolved_client,
        config=config,
        key_value_cache=cache,
        idempotency_store=RedisIdempotencyStore(resolved_client),
        rate_limiter=RedisDistributedRateLimiter(resolved_client),
        execution_semaphore=RedisExecutionSemaphore(
            client=resolved_client,
            ttl_seconds=execution_semaphore_ttl_seconds,
        ),
    )


def create_redis_key_value_cache(
    *,
    url: Optional[str] = None,
    db: Optional[int] = None,
    key_prefix: Optional[str] = None,
    client: Optional["redis.Redis"] = None,
    **config_overrides: object,
) -> KeyValueCache:
    """Catalog factory for ``"redis"`` / ``KEY_VALUE_CACHE``."""
    return create_redis_integration(
        url=url,
        db=db,
        key_prefix=key_prefix,
        client=client,
        **config_overrides,
    ).key_value_cache


def create_redis_kv_store(
    *,
    url: Optional[str] = None,
    db: Optional[int] = None,
    key_prefix: Optional[str] = None,
    client: Optional["redis.Redis"] = None,
    **config_overrides: object,
) -> DistributedKVStore:
    """``DistributedKVStore`` for queueing transport wiring."""
    return create_redis_integration(
        url=url,
        db=db,
        key_prefix=key_prefix,
        client=client,
        **config_overrides,
    ).kv_store


def create_redis_idempotency_store(
    *,
    url: Optional[str] = None,
    db: Optional[int] = None,
    client: Optional["redis.Redis"] = None,
    **config_overrides: object,
) -> IdempotencyStore:
    return create_redis_integration(
        url=url,
        db=db,
        client=client,
        **config_overrides,
    ).idempotency_store


def create_redis_rate_limiter(
    *,
    url: Optional[str] = None,
    db: Optional[int] = None,
    client: Optional["redis.Redis"] = None,
    **config_overrides: object,
) -> DistributedRateLimiter:
    return create_redis_integration(
        url=url,
        db=db,
        client=client,
        **config_overrides,
    ).rate_limiter


def create_redis_execution_semaphore(
    *,
    url: Optional[str] = None,
    db: Optional[int] = None,
    client: Optional["redis.Redis"] = None,
    execution_semaphore_ttl_seconds: int = 300,
    **config_overrides: object,
) -> DistributedExecutionSemaphore:
    return create_redis_integration(
        url=url,
        db=db,
        client=client,
        execution_semaphore_ttl_seconds=execution_semaphore_ttl_seconds,
        **config_overrides,
    ).execution_semaphore


def create_redis_rerank_cache(
    *,
    url: Optional[str] = None,
    db: Optional[int] = None,
    client: Optional["redis.Redis"] = None,
    ttl_seconds: int = 3600,
    key_prefix: str = "rerank",
    **config_overrides: object,
) -> BaseRerankCache:
    """RAG rerank cache — uses the shared Redis client factory."""
    resolved_client = create_redis_client(
        url=url,
        db=db,
        client=client,
        **config_overrides,
    )
    return RedisRerankCache(
        redis_client=resolved_client,
        ttl_seconds=ttl_seconds,
        key_prefix=key_prefix,
    )

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.key_value_cache.redis.integration import (
    REDIS_KEY_VALUE_CACHE_PROVIDER_ID,
    RedisKeyValueCacheIntegration,
    RedisKeyValueCacheIntegrationConfig,
    RedisKeyValueCacheClient,
)


def create_redis_key_value_cache_integration(
    *,
    client: RedisKeyValueCacheIntegrationClient | None = None,
    enabled: bool = False,
) -> RedisKeyValueCacheIntegration:
    """
    Build a contract-based Redis key value cache integration.

    Compatibility shim — constructs Integration via from_store (create_redis_integration) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Redis key value cache integration requires an injected client when enabled=True",
        )
    if client is not None:
        return RedisKeyValueCacheIntegration.from_client(client, enabled=enabled)
    return RedisKeyValueCacheIntegration.for_provider(
        provider_id=REDIS_KEY_VALUE_CACHE_PROVIDER_ID,
        display_name="Redis",
        config=RedisKeyValueCacheIntegrationConfig(enabled=enabled),
    )

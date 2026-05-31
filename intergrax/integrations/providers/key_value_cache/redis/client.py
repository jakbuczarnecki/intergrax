# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared Redis client factory — single connection entry for the redis integration."""

from __future__ import annotations

from typing import Optional

import redis

from intergrax.integrations.providers.key_value_cache.redis.config import RedisIntegrationConfig


def resolve_redis_config(**overrides: object) -> RedisIntegrationConfig:
    """Build config from env with optional overrides (url, db, key_prefix, …)."""
    return RedisIntegrationConfig.from_env(**overrides)


def create_redis_client(
    *,
    url: Optional[str] = None,
    db: Optional[int] = None,
    client: Optional[redis.Redis] = None,
    **config_overrides: object,
) -> redis.Redis:
    """
    Create or reuse a ``redis.Redis`` client for all redis integration facades.

    Inject ``client`` in tests; production uses ``INTERGRAX_REDIS_*`` env vars.
    """
    if client is not None:
        return client

    overrides: dict[str, object] = dict(config_overrides)
    if url is not None:
        overrides["url"] = url
    if db is not None:
        overrides["db"] = db

    config = resolve_redis_config(**overrides)
    return redis.Redis.from_url(
        config.url,
        db=config.db,
        decode_responses=config.decode_responses,
    )

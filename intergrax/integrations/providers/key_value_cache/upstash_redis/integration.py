# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Upstash Redis key value cache integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import KeyValueCacheIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

UPSTASH_REDIS_KEY_VALUE_CACHE_PROVIDER_ID = "upstash_redis"


class UpstashRedisKeyValueCacheIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Upstash Redis key value cache integration."""

    pass


@runtime_checkable
class UpstashRedisKeyValueCacheClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class UpstashRedisKeyValueCacheIntegration(KeyValueCacheIntegrationContract):
    """
    Upstash Redis key value cache integration.

    The legacy facade (create_upstash_redis_key_value_cache) remains separate and backward-compatible.
    """

    config: UpstashRedisKeyValueCacheIntegrationConfig = UpstashRedisKeyValueCacheIntegrationConfig()
    _client: UpstashRedisKeyValueCacheClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: UpstashRedisKeyValueCacheClient,
        *,
        enabled: bool = False,
    ) -> UpstashRedisKeyValueCacheIntegration:
        integration = cls.for_provider(
            provider_id=UPSTASH_REDIS_KEY_VALUE_CACHE_PROVIDER_ID,
            display_name="Upstash Redis",
            config=UpstashRedisKeyValueCacheIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> UpstashRedisKeyValueCacheClient | None:
        return self._client

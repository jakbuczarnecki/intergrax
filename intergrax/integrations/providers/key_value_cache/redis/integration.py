# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Redis key value cache integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import KeyValueCacheIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

REDIS_KEY_VALUE_CACHE_PROVIDER_ID = "redis"


class RedisKeyValueCacheIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Redis key value cache integration."""

    pass


@runtime_checkable
class RedisKeyValueCacheClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class RedisKeyValueCacheIntegration(KeyValueCacheIntegrationContract):
    """
    Redis key value cache integration.

    The legacy facade (create_redis_integration) remains separate and backward-compatible.
    """

    config: RedisKeyValueCacheIntegrationConfig = RedisKeyValueCacheIntegrationConfig()
    _client: RedisKeyValueCacheClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: RedisKeyValueCacheClient,
        *,
        enabled: bool = False,
    ) -> RedisKeyValueCacheIntegration:
        integration = cls.for_provider(
            provider_id=REDIS_KEY_VALUE_CACHE_PROVIDER_ID,
            display_name="Redis",
            config=RedisKeyValueCacheIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> RedisKeyValueCacheClient | None:
        return self._client

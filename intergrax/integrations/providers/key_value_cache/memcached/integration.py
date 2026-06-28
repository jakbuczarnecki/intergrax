# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Memcached key value cache integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import KeyValueCacheIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

MEMCACHED_KEY_VALUE_CACHE_PROVIDER_ID = "memcached"


class MemcachedKeyValueCacheIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Memcached key value cache integration."""

    pass


@runtime_checkable
class MemcachedKeyValueCacheClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class MemcachedKeyValueCacheIntegration(KeyValueCacheIntegrationContract):
    """
    Memcached key value cache integration.

    The legacy facade (create_memcached_key_value_cache) remains separate and backward-compatible.
    """

    config: MemcachedKeyValueCacheIntegrationConfig = MemcachedKeyValueCacheIntegrationConfig()
    _client: MemcachedKeyValueCacheClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: MemcachedKeyValueCacheClient,
        *,
        enabled: bool = False,
    ) -> MemcachedKeyValueCacheIntegration:
        integration = cls.for_provider(
            provider_id=MEMCACHED_KEY_VALUE_CACHE_PROVIDER_ID,
            display_name="Memcached",
            config=MemcachedKeyValueCacheIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> MemcachedKeyValueCacheClient | None:
        return self._client

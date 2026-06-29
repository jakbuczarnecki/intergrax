# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Memcached key value cache integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.key_value_cache import KeyValueCache
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
    Single public Memcached key value cache entrypoint.

    Legacy catalog factory (create_memcached_key_value_cache) delegates to this class.
    """

    config: MemcachedKeyValueCacheIntegrationConfig = MemcachedKeyValueCacheIntegrationConfig()
    _client: MemcachedKeyValueCacheClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> MemcachedKeyValueCacheIntegration:
        integration = cls.for_provider(
            provider_id=MEMCACHED_KEY_VALUE_CACHE_PROVIDER_ID,
            display_name="Memcached",
            config=MemcachedKeyValueCacheIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Memcached integration requires a runtime delegate")
        return self._runtime



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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

KeyValueCache.register(MemcachedKeyValueCacheIntegration)

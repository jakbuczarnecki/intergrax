# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Optional

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.key_value_cache import KeyValueCache
from intergrax.integrations.examples.custom_memory_kv.adapter import InProcessKeyValueCache
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig
from intergrax.runtime.integrations.categories.data import KeyValueCacheIntegrationContract

CUSTOM_MEMORY_KV_PROVIDER_ID = "custom_memory_kv"


class CustomMemoryKvIntegrationConfig(CategoryIntegrationConfig):
    pass


class CustomMemoryKvIntegration(KeyValueCacheIntegrationContract):
    config: CustomMemoryKvIntegrationConfig = CustomMemoryKvIntegrationConfig()
    _client: KeyValueCache | None = PrivateAttr(default=None)

    def delete(self, tenant_id, key):
        return self._require_client().delete(tenant_id, key)

    def get(self, tenant_id, key):
        return self._require_client().get(tenant_id, key)

    def set(self, tenant_id, key, value, ttl_seconds: Optional[int] = None):
        return self._require_client().set(tenant_id, key, value, ttl_seconds=ttl_seconds)

    def set_if_absent(self, tenant_id, key, value, ttl_seconds: Optional[int] = None):
        return self._require_client().set_if_absent(tenant_id, key, value, ttl_seconds=ttl_seconds)

    def _require_client(self) -> KeyValueCache:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client

    @classmethod
    def from_client(
        cls,
        client: KeyValueCache,
        *,
        enabled: bool = False,
    ) -> CustomMemoryKvIntegration:
        integration = cls.for_provider(
            provider_id=CUSTOM_MEMORY_KV_PROVIDER_ID,
            display_name="Custom memory KV",
            config=CustomMemoryKvIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration


KeyValueCache.register(CustomMemoryKvIntegration)

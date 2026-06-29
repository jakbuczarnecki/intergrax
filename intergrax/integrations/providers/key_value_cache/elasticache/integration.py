# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Elasticache key value cache integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.key_value_cache import KeyValueCache
from intergrax.runtime.integrations.categories.data import KeyValueCacheIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

ELASTICACHE_KEY_VALUE_CACHE_PROVIDER_ID = "elasticache"


class ElasticacheKeyValueCacheIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Elasticache key value cache integration."""

    pass


ElasticacheKeyValueCacheClient = KeyValueCache

class ElasticacheKeyValueCacheIntegration(KeyValueCacheIntegrationContract):
    """
    Single public Elasticache key value cache entrypoint.

    Legacy catalog factory (create_elasticache_key_value_cache) owns catalog behavior; legacy factories use from_client().
    """

    config: ElasticacheKeyValueCacheIntegrationConfig = ElasticacheKeyValueCacheIntegrationConfig()
    _client: ElasticacheKeyValueCacheClient | None = PrivateAttr(default=None)
    

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
        client: ElasticacheKeyValueCacheClient,
        *,
        enabled: bool = False,
    ) -> ElasticacheKeyValueCacheIntegration:
        integration = cls.for_provider(
            provider_id=ELASTICACHE_KEY_VALUE_CACHE_PROVIDER_ID,
            display_name="Elasticache",
            config=ElasticacheKeyValueCacheIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> ElasticacheKeyValueCacheClient | None:
        return self._client

KeyValueCache.register(ElasticacheKeyValueCacheIntegration)

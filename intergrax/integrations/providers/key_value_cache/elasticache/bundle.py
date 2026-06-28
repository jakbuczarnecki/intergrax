# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_elasticache_key_value_cache

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.key_value_cache.elasticache.integration import (
    ELASTICACHE_KEY_VALUE_CACHE_PROVIDER_ID,
    ElasticacheKeyValueCacheIntegration,
    ElasticacheKeyValueCacheIntegrationConfig,
    ElasticacheKeyValueCacheClient,
)

__all__ = [
    "create_elasticache_key_value_cache",
    "create_elasticache_key_value_cache_integration",
]


def create_elasticache_key_value_cache_integration(
    *,
    client: ElasticacheKeyValueCacheClient | None = None,
    enabled: bool = False,
) -> ElasticacheKeyValueCacheIntegration:
    """
    Build a contract-based Elasticache key value cache integration.

    The legacy facade (create_elasticache_key_value_cache) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Elasticache key value cache integration requires an injected client when enabled=True",
        )
    if client is not None:
        return ElasticacheKeyValueCacheIntegration.from_client(client, enabled=enabled)
    return ElasticacheKeyValueCacheIntegration.for_provider(
        provider_id=ELASTICACHE_KEY_VALUE_CACHE_PROVIDER_ID,
        display_name="Elasticache",
        config=ElasticacheKeyValueCacheIntegrationConfig(enabled=enabled),
    )

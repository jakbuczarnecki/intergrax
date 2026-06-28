# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Elasticache key value cache integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import KeyValueCacheIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

ELASTICACHE_KEY_VALUE_CACHE_PROVIDER_ID = "elasticache"


class ElasticacheKeyValueCacheIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Elasticache key value cache integration."""

    pass


@runtime_checkable
class ElasticacheKeyValueCacheClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class ElasticacheKeyValueCacheIntegration(KeyValueCacheIntegrationContract):
    """
    Elasticache key value cache integration.

    The legacy facade (create_elasticache_key_value_cache) remains separate and backward-compatible.
    """

    config: ElasticacheKeyValueCacheIntegrationConfig = ElasticacheKeyValueCacheIntegrationConfig()
    _client: ElasticacheKeyValueCacheClient | None = PrivateAttr(default=None)

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

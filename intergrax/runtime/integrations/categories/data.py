# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Data-store provider category contracts (INTEGRATIONS-2A)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from intergrax.runtime.integrations.categories._base import (
    CategoryIntegrationConfig,
    _CONNECT_READ_HEALTH,
    _CONNECT_READ_WRITE_HEALTH,
    category_for_provider,
)
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationContract,
    PlatformIntegrationKind,
)

RELATIONAL_STORE_INTEGRATION_CONTRACT_SCHEMA = "relational_store_integration_contract.v1"
KEY_VALUE_CACHE_INTEGRATION_CONTRACT_SCHEMA = "key_value_cache_integration_contract.v1"
GRAPH_STORE_INTEGRATION_CONTRACT_SCHEMA = "graph_store_integration_contract.v1"


class RelationalStoreIntegrationContract(PlatformIntegrationContract):
    """Category contract for relational_store providers (sqlite, postgres, …)."""

    schema_id: Literal["relational_store_integration_contract.v1"] = (
        RELATIONAL_STORE_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.RELATIONAL_STORE.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_READ_WRITE_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> RelationalStoreIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.RELATIONAL_STORE.value,
            default_capabilities=_CONNECT_READ_WRITE_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )


class KeyValueCacheIntegrationContract(PlatformIntegrationContract):
    """Category contract for key_value_cache providers (redis, memcached, …)."""

    schema_id: Literal["key_value_cache_integration_contract.v1"] = (
        KEY_VALUE_CACHE_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.KEY_VALUE_CACHE.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_READ_WRITE_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> KeyValueCacheIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.KEY_VALUE_CACHE.value,
            default_capabilities=_CONNECT_READ_WRITE_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )


class GraphStoreIntegrationContract(PlatformIntegrationContract):
    """Category contract for graph_store providers (neo4j, memgraph, …)."""

    schema_id: Literal["graph_store_integration_contract.v1"] = GRAPH_STORE_INTEGRATION_CONTRACT_SCHEMA
    integration_kind: str = PlatformIntegrationKind.GRAPH_STORE.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_READ_WRITE_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> GraphStoreIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.GRAPH_STORE.value,
            default_capabilities=_CONNECT_READ_WRITE_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )

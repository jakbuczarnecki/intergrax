# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Search and rerank provider category contracts (INTEGRATIONS-2A)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from intergrax.runtime.integrations.categories._base import (
    CategoryIntegrationConfig,
    _CONNECT_READ_HEALTH,
    category_for_provider,
)
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationContract,
    PlatformIntegrationKind,
)

SEARCH_PROVIDER_INTEGRATION_CONTRACT_SCHEMA = "search_provider_integration_contract.v1"
RERANK_PROVIDER_INTEGRATION_CONTRACT_SCHEMA = "rerank_provider_integration_contract.v1"


class SearchProviderIntegrationContract(PlatformIntegrationContract):
    """Category contract for search_provider slugs (google_cse, tavily, …)."""

    schema_id: Literal["search_provider_integration_contract.v1"] = (
        SEARCH_PROVIDER_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.SEARCH_PROVIDER.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_READ_HEALTH
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
    ) -> SearchProviderIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.SEARCH_PROVIDER.value,
            default_capabilities=_CONNECT_READ_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )


class RerankProviderIntegrationContract(PlatformIntegrationContract):
    """Category contract for rerank_provider slugs (cohere_rerank, jina_rerank, …)."""

    schema_id: Literal["rerank_provider_integration_contract.v1"] = (
        RERANK_PROVIDER_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.RERANK_PROVIDER.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_READ_HEALTH
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
    ) -> RerankProviderIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.RERANK_PROVIDER.value,
            default_capabilities=_CONNECT_READ_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )

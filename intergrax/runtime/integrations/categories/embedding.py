# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Embedding provider category contract (P2-002-B1)."""

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

EMBEDDING_PROVIDER_INTEGRATION_CONTRACT_SCHEMA = "embedding_provider_integration_contract.v1"


class EmbeddingProviderIntegrationContract(PlatformIntegrationContract):
    """Category contract for embedding_provider slugs (Integrations catalog boundary only)."""

    schema_id: Literal["embedding_provider_integration_contract.v1"] = (
        EMBEDDING_PROVIDER_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.EMBEDDING_PROVIDER.value
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
    ) -> EmbeddingProviderIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.EMBEDDING_PROVIDER.value,
            default_capabilities=_CONNECT_READ_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )

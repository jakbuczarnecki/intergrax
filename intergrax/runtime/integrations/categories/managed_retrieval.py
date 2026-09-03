# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Managed retrieval provider category contract."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from intergrax.runtime.integrations.categories._base import (
    CategoryIntegrationConfig,
    _CONNECT_READ_WRITE_HEALTH,
    category_for_provider,
)
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationContract,
    PlatformIntegrationKind,
)

MANAGED_RETRIEVAL_INTEGRATION_CONTRACT_SCHEMA = "managed_retrieval_integration_contract.v1"


class ManagedRetrievalIntegrationContract(PlatformIntegrationContract):
    """Category contract for managed_retrieval slugs (openai, …)."""

    schema_id: Literal["managed_retrieval_integration_contract.v1"] = (
        MANAGED_RETRIEVAL_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.MANAGED_RETRIEVAL.value
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
    ) -> ManagedRetrievalIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.MANAGED_RETRIEVAL.value,
            default_capabilities=_CONNECT_READ_WRITE_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Object and vector storage provider category contracts (INTEGRATIONS-2A)."""

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

OBJECT_STORAGE_INTEGRATION_CONTRACT_SCHEMA = "object_storage_integration_contract.v1"
VECTOR_STORE_INTEGRATION_CONTRACT_SCHEMA = "vector_store_integration_contract.v1"


class ObjectStorageIntegrationContract(PlatformIntegrationContract):
    """Category contract for object_storage providers (s3, gcs, minio, …)."""

    schema_id: Literal["object_storage_integration_contract.v1"] = (
        OBJECT_STORAGE_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.OBJECT_STORAGE.value
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
    ) -> ObjectStorageIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.OBJECT_STORAGE.value,
            default_capabilities=_CONNECT_READ_WRITE_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )


class VectorStoreIntegrationContract(PlatformIntegrationContract):
    """Category contract for vector_store providers (pinecone, qdrant, weaviate, …)."""

    schema_id: Literal["vector_store_integration_contract.v1"] = VECTOR_STORE_INTEGRATION_CONTRACT_SCHEMA
    integration_kind: str = PlatformIntegrationKind.VECTOR_STORE.value
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
    ) -> VectorStoreIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.VECTOR_STORE.value,
            default_capabilities=_CONNECT_READ_WRITE_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )

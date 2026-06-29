# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Callable, Optional

from intergrax.integrations.contracts.object_storage import ObjectStorage
from intergrax.integrations.providers.object_storage.azure_blob.config import AzureBlobIntegrationConfig
from intergrax.integrations.providers.object_storage.azure_blob.opens import open_azure_blob_object_storage


def resolve_azure_blob_config(**overrides: object) -> AzureBlobIntegrationConfig:
    return AzureBlobIntegrationConfig.from_env(**overrides)


def create_azure_blob_object_storage(
    *,
    object_storage: Optional[ObjectStorage] = None,
    container_client: Optional[object] = None,
    container_client_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> ObjectStorage:
    config = resolve_azure_blob_config(**config_overrides)
    return open_azure_blob_object_storage(
        config,
        implementation=object_storage,
        container_client=container_client,
        container_client_factory=container_client_factory,
    )

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.object_storage.azure_blob.integration import (
    AZURE_BLOB_OBJECT_STORAGE_PROVIDER_ID,
    AzureBlobObjectStorageIntegration,
    AzureBlobObjectStorageIntegrationConfig,
    AzureBlobObjectStorageClient,
)


def create_azure_blob_object_storage_integration(
    *,
    client: AzureBlobObjectStorageClient | None = None,
    enabled: bool = False,
) -> AzureBlobObjectStorageIntegration:
    """
    Build a contract-based Azure Blob object storage integration.

    The legacy facade (create_azure_blob_object_storage) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Azure Blob object storage integration requires an injected client when enabled=True",
        )
    if client is not None:
        return AzureBlobObjectStorageIntegration.from_client(client, enabled=enabled)
    return AzureBlobObjectStorageIntegration.for_provider(
        provider_id=AZURE_BLOB_OBJECT_STORAGE_PROVIDER_ID,
        display_name="Azure Blob",
        config=AzureBlobObjectStorageIntegrationConfig(enabled=enabled),
    )

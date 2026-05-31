# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure Blob opens — only place that imports azure-storage-blob."""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.object_storage import ObjectStorage
from intergrax.integrations.providers.object_storage.azure_blob.client import build_azure_blob_object_storage
from intergrax.integrations.providers.object_storage.azure_blob.config import AzureBlobIntegrationConfig


def _import_blob_service_client() -> Any:
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError as exc:
        raise IntegrationConfigurationError(
            "Azure Blob integration requires azure-storage-blob. "
            "Install with: uv pip install azure-storage-blob"
        ) from exc
    return BlobServiceClient


def open_azure_blob_object_storage(
    config: AzureBlobIntegrationConfig,
    *,
    implementation: Optional[ObjectStorage] = None,
    container_client: Optional[Any] = None,
    container_client_factory: Optional[Callable[[], Any]] = None,
) -> ObjectStorage:
    if implementation is not None:
        return implementation
    if container_client is not None:
        return build_azure_blob_object_storage(config, container_client)
    if container_client_factory is not None:
        return build_azure_blob_object_storage(config, container_client_factory())
    BlobServiceClient = _import_blob_service_client()
    if config.connection_string:
        service = BlobServiceClient.from_connection_string(config.connection_string)
    elif config.account_url:
        service = BlobServiceClient(account_url=config.account_url)
    else:
        raise IntegrationConfigurationError(
            "Azure Blob requires INTERGRAX_AZURE_BLOB_CONNECTION_STRING or INTERGRAX_AZURE_BLOB_ACCOUNT_URL"
        )
    container = service.get_container_client(config.require_container())
    return build_azure_blob_object_storage(config, container)

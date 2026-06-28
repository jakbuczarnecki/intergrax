# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure Blob object storage integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.storage import ObjectStorageIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

AZURE_BLOB_OBJECT_STORAGE_PROVIDER_ID = "azure_blob"


class AzureBlobObjectStorageIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Azure Blob object storage integration."""

    pass


@runtime_checkable
class AzureBlobObjectStorageClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class AzureBlobObjectStorageIntegration(ObjectStorageIntegrationContract):
    """
    Azure Blob object storage integration.

    The legacy facade (create_azure_blob_object_storage) remains separate and backward-compatible.
    """

    config: AzureBlobObjectStorageIntegrationConfig = AzureBlobObjectStorageIntegrationConfig()
    _client: AzureBlobObjectStorageClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: AzureBlobObjectStorageClient,
        *,
        enabled: bool = False,
    ) -> AzureBlobObjectStorageIntegration:
        integration = cls.for_provider(
            provider_id=AZURE_BLOB_OBJECT_STORAGE_PROVIDER_ID,
            display_name="Azure Blob",
            config=AzureBlobObjectStorageIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> AzureBlobObjectStorageClient | None:
        return self._client

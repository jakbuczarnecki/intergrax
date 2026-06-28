# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Drive object storage integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.storage import ObjectStorageIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GOOGLE_DRIVE_OBJECT_STORAGE_PROVIDER_ID = "google_drive"


class GoogleDriveObjectStorageIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Google Drive object storage integration."""

    pass


@runtime_checkable
class GoogleDriveObjectStorageClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class GoogleDriveObjectStorageIntegration(ObjectStorageIntegrationContract):
    """
    Google Drive object storage integration.

    The legacy facade (create_google_drive_object_storage) remains separate and backward-compatible.
    """

    config: GoogleDriveObjectStorageIntegrationConfig = GoogleDriveObjectStorageIntegrationConfig()
    _client: GoogleDriveObjectStorageClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: GoogleDriveObjectStorageClient,
        *,
        enabled: bool = False,
    ) -> GoogleDriveObjectStorageIntegration:
        integration = cls.for_provider(
            provider_id=GOOGLE_DRIVE_OBJECT_STORAGE_PROVIDER_ID,
            display_name="Google Drive",
            config=GoogleDriveObjectStorageIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> GoogleDriveObjectStorageClient | None:
        return self._client

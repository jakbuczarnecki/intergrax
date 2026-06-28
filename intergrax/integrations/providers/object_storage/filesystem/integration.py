# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Filesystem object storage integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.storage import ObjectStorageIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

FILESYSTEM_OBJECT_STORAGE_PROVIDER_ID = "filesystem"


class FilesystemObjectStorageIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Filesystem object storage integration."""

    pass


@runtime_checkable
class FilesystemObjectStorageClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class FilesystemObjectStorageIntegration(ObjectStorageIntegrationContract):
    """
    Filesystem object storage integration.

    The legacy facade (create_filesystem_object_storage) remains separate and backward-compatible.
    """

    config: FilesystemObjectStorageIntegrationConfig = FilesystemObjectStorageIntegrationConfig()
    _client: FilesystemObjectStorageClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: FilesystemObjectStorageClient,
        *,
        enabled: bool = False,
    ) -> FilesystemObjectStorageIntegration:
        integration = cls.for_provider(
            provider_id=FILESYSTEM_OBJECT_STORAGE_PROVIDER_ID,
            display_name="Filesystem",
            config=FilesystemObjectStorageIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> FilesystemObjectStorageClient | None:
        return self._client

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Minio object storage integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.storage import ObjectStorageIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

MINIO_OBJECT_STORAGE_PROVIDER_ID = "minio"


class MinioObjectStorageIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Minio object storage integration."""

    pass


@runtime_checkable
class MinioObjectStorageClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class MinioObjectStorageIntegration(ObjectStorageIntegrationContract):
    """
    Minio object storage integration.

    The legacy facade (create_minio_object_storage) remains separate and backward-compatible.
    """

    config: MinioObjectStorageIntegrationConfig = MinioObjectStorageIntegrationConfig()
    _client: MinioObjectStorageClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: MinioObjectStorageClient,
        *,
        enabled: bool = False,
    ) -> MinioObjectStorageIntegration:
        integration = cls.for_provider(
            provider_id=MINIO_OBJECT_STORAGE_PROVIDER_ID,
            display_name="Minio",
            config=MinioObjectStorageIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> MinioObjectStorageClient | None:
        return self._client

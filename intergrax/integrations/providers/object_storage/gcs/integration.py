# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Gcs object storage integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.storage import ObjectStorageIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GCS_OBJECT_STORAGE_PROVIDER_ID = "gcs"


class GcsObjectStorageIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Gcs object storage integration."""

    pass


@runtime_checkable
class GcsObjectStorageClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class GcsObjectStorageIntegration(ObjectStorageIntegrationContract):
    """
    Gcs object storage integration.

    The legacy facade (create_gcs_object_storage) remains separate and backward-compatible.
    """

    config: GcsObjectStorageIntegrationConfig = GcsObjectStorageIntegrationConfig()
    _client: GcsObjectStorageClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: GcsObjectStorageClient,
        *,
        enabled: bool = False,
    ) -> GcsObjectStorageIntegration:
        integration = cls.for_provider(
            provider_id=GCS_OBJECT_STORAGE_PROVIDER_ID,
            display_name="Gcs",
            config=GcsObjectStorageIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> GcsObjectStorageClient | None:
        return self._client

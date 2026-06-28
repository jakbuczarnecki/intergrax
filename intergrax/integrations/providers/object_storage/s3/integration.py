# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""S3 object storage integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.storage import ObjectStorageIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

S3_OBJECT_STORAGE_PROVIDER_ID = "s3"


class S3ObjectStorageIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for S3 object storage integration."""

    pass


@runtime_checkable
class S3ObjectStorageClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class S3ObjectStorageIntegration(ObjectStorageIntegrationContract):
    """
    S3 object storage integration.

    The legacy facade (create_s3_integration) remains separate and backward-compatible.
    """

    config: S3ObjectStorageIntegrationConfig = S3ObjectStorageIntegrationConfig()
    _client: S3ObjectStorageClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: S3ObjectStorageClient,
        *,
        enabled: bool = False,
    ) -> S3ObjectStorageIntegration:
        integration = cls.for_provider(
            provider_id=S3_OBJECT_STORAGE_PROVIDER_ID,
            display_name="S3",
            config=S3ObjectStorageIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> S3ObjectStorageClient | None:
        return self._client

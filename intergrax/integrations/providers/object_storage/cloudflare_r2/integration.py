# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cloudflare R2 object storage integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.storage import ObjectStorageIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

CLOUDFLARE_R2_OBJECT_STORAGE_PROVIDER_ID = "cloudflare_r2"


class CloudflareR2ObjectStorageIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Cloudflare R2 object storage integration."""

    pass


@runtime_checkable
class CloudflareR2ObjectStorageClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class CloudflareR2ObjectStorageIntegration(ObjectStorageIntegrationContract):
    """
    Cloudflare R2 object storage integration.

    The legacy facade (create_cloudflare_r2_object_storage) remains separate and backward-compatible.
    """

    config: CloudflareR2ObjectStorageIntegrationConfig = CloudflareR2ObjectStorageIntegrationConfig()
    _client: CloudflareR2ObjectStorageClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: CloudflareR2ObjectStorageClient,
        *,
        enabled: bool = False,
    ) -> CloudflareR2ObjectStorageIntegration:
        integration = cls.for_provider(
            provider_id=CLOUDFLARE_R2_OBJECT_STORAGE_PROVIDER_ID,
            display_name="Cloudflare R2",
            config=CloudflareR2ObjectStorageIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> CloudflareR2ObjectStorageClient | None:
        return self._client

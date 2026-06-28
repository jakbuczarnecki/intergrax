# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Backblaze B2 object storage integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.storage import ObjectStorageIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

BACKBLAZE_B2_OBJECT_STORAGE_PROVIDER_ID = "backblaze_b2"


class BackblazeB2ObjectStorageIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Backblaze B2 object storage integration."""

    pass


@runtime_checkable
class BackblazeB2ObjectStorageClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class BackblazeB2ObjectStorageIntegration(ObjectStorageIntegrationContract):
    """
    Backblaze B2 object storage integration.

    The legacy facade (create_backblaze_b2_object_storage) remains separate and backward-compatible.
    """

    config: BackblazeB2ObjectStorageIntegrationConfig = BackblazeB2ObjectStorageIntegrationConfig()
    _client: BackblazeB2ObjectStorageClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: BackblazeB2ObjectStorageClient,
        *,
        enabled: bool = False,
    ) -> BackblazeB2ObjectStorageIntegration:
        integration = cls.for_provider(
            provider_id=BACKBLAZE_B2_OBJECT_STORAGE_PROVIDER_ID,
            display_name="Backblaze B2",
            config=BackblazeB2ObjectStorageIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> BackblazeB2ObjectStorageClient | None:
        return self._client

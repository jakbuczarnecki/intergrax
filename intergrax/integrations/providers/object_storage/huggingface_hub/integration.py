# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Huggingface Hub object storage integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.storage import ObjectStorageIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

HUGGINGFACE_HUB_OBJECT_STORAGE_PROVIDER_ID = "huggingface_hub"


class HuggingfaceHubObjectStorageIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Huggingface Hub object storage integration."""

    pass


@runtime_checkable
class HuggingfaceHubObjectStorageClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class HuggingfaceHubObjectStorageIntegration(ObjectStorageIntegrationContract):
    """
    Huggingface Hub object storage integration.

    The legacy facade (create_huggingface_hub_object_storage) remains separate and backward-compatible.
    """

    config: HuggingfaceHubObjectStorageIntegrationConfig = HuggingfaceHubObjectStorageIntegrationConfig()
    _client: HuggingfaceHubObjectStorageClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: HuggingfaceHubObjectStorageClient,
        *,
        enabled: bool = False,
    ) -> HuggingfaceHubObjectStorageIntegration:
        integration = cls.for_provider(
            provider_id=HUGGINGFACE_HUB_OBJECT_STORAGE_PROVIDER_ID,
            display_name="Huggingface Hub",
            config=HuggingfaceHubObjectStorageIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> HuggingfaceHubObjectStorageClient | None:
        return self._client

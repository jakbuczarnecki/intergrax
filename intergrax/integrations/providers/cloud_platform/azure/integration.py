# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure cloud platform integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.devops import CloudPlatformIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

AZURE_CLOUD_PLATFORM_PROVIDER_ID = "azure"


class AzureCloudPlatformIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Azure cloud platform integration."""

    pass


@runtime_checkable
class AzureCloudPlatformClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class AzureCloudPlatformIntegration(CloudPlatformIntegrationContract):
    """
    Azure cloud platform integration.

    The legacy facade (create_azure_integration) remains separate and backward-compatible.
    """

    config: AzureCloudPlatformIntegrationConfig = AzureCloudPlatformIntegrationConfig()
    _client: AzureCloudPlatformClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: AzureCloudPlatformClient,
        *,
        enabled: bool = False,
    ) -> AzureCloudPlatformIntegration:
        integration = cls.for_provider(
            provider_id=AZURE_CLOUD_PLATFORM_PROVIDER_ID,
            display_name="Azure",
            config=AzureCloudPlatformIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> AzureCloudPlatformClient | None:
        return self._client

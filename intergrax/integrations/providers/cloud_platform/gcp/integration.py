# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""GCP cloud platform integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.devops import CloudPlatformIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GCP_CLOUD_PLATFORM_PROVIDER_ID = "gcp"


class GcpCloudPlatformIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for GCP cloud platform integration."""

    pass


@runtime_checkable
class GcpCloudPlatformClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class GcpCloudPlatformIntegration(CloudPlatformIntegrationContract):
    """
    GCP cloud platform integration.

    The legacy facade (create_gcp_integration) remains separate and backward-compatible.
    """

    config: GcpCloudPlatformIntegrationConfig = GcpCloudPlatformIntegrationConfig()
    _client: GcpCloudPlatformClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: GcpCloudPlatformClient,
        *,
        enabled: bool = False,
    ) -> GcpCloudPlatformIntegration:
        integration = cls.for_provider(
            provider_id=GCP_CLOUD_PLATFORM_PROVIDER_ID,
            display_name="GCP",
            config=GcpCloudPlatformIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> GcpCloudPlatformClient | None:
        return self._client

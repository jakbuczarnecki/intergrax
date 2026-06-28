# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Localstack cloud platform integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.devops import CloudPlatformIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

LOCALSTACK_CLOUD_PLATFORM_PROVIDER_ID = "localstack"


class LocalstackCloudPlatformIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Localstack cloud platform integration."""

    pass


@runtime_checkable
class LocalstackCloudPlatformClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class LocalstackCloudPlatformIntegration(CloudPlatformIntegrationContract):
    """
    Localstack cloud platform integration.

    The legacy facade (create_localstack_cloud_platform) remains separate and backward-compatible.
    """

    config: LocalstackCloudPlatformIntegrationConfig = LocalstackCloudPlatformIntegrationConfig()
    _client: LocalstackCloudPlatformClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: LocalstackCloudPlatformClient,
        *,
        enabled: bool = False,
    ) -> LocalstackCloudPlatformIntegration:
        integration = cls.for_provider(
            provider_id=LOCALSTACK_CLOUD_PLATFORM_PROVIDER_ID,
            display_name="Localstack",
            config=LocalstackCloudPlatformIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> LocalstackCloudPlatformClient | None:
        return self._client

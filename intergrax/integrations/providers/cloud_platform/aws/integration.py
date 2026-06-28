# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""AWS cloud platform integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.devops import CloudPlatformIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

AWS_CLOUD_PLATFORM_PROVIDER_ID = "aws"


class AwsCloudPlatformIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for AWS cloud platform integration."""

    pass


@runtime_checkable
class AwsCloudPlatformClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class AwsCloudPlatformIntegration(CloudPlatformIntegrationContract):
    """
    AWS cloud platform integration.

    The legacy facade (create_aws_integration) remains separate and backward-compatible.
    """

    config: AwsCloudPlatformIntegrationConfig = AwsCloudPlatformIntegrationConfig()
    _client: AwsCloudPlatformClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: AwsCloudPlatformClient,
        *,
        enabled: bool = False,
    ) -> AwsCloudPlatformIntegration:
        integration = cls.for_provider(
            provider_id=AWS_CLOUD_PLATFORM_PROVIDER_ID,
            display_name="AWS",
            config=AwsCloudPlatformIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> AwsCloudPlatformClient | None:
        return self._client

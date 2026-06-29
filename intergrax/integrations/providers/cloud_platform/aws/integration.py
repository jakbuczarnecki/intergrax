# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Aws cloud platform integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.runtime.integrations.categories.devops import CloudPlatformIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

AWS_CLOUD_PLATFORM_PROVIDER_ID = "aws"


class AwsCloudPlatformIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Aws cloud platform integration."""

    pass


AwsCloudPlatformClient = CloudPlatform

class AwsCloudPlatformIntegration(CloudPlatformIntegrationContract):
    """
    Single public Aws cloud platform entrypoint.

    Legacy catalog factory (create_aws_integration) owns catalog behavior; legacy factories use from_client().
    """

    config: AwsCloudPlatformIntegrationConfig = AwsCloudPlatformIntegrationConfig()
    _client: AwsCloudPlatformClient | None = PrivateAttr(default=None)
    

    @property
    def default_region(self):
        return getattr(self._require_client(), 'default_region')

    def health(self):
        return self._require_client().health()

    def resolve(self, category):
        return self._require_client().resolve(category)

    @property
    def slug(self):
        return getattr(self._require_client(), 'slug')

    def _require_client(self) -> CloudPlatform:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: AwsCloudPlatformClient,
        *,
        enabled: bool = False,
    ) -> AwsCloudPlatformIntegration:
        integration = cls.for_provider(
            provider_id=AWS_CLOUD_PLATFORM_PROVIDER_ID,
            display_name="Aws",
            config=AwsCloudPlatformIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> AwsCloudPlatformClient | None:
        return self._client

CloudPlatform.register(AwsCloudPlatformIntegration)

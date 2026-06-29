# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure cloud platform integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.runtime.integrations.categories.devops import CloudPlatformIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig
from intergrax.utils import attribute_access


AZURE_CLOUD_PLATFORM_PROVIDER_ID = "azure"


class AzureCloudPlatformIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Azure cloud platform integration."""

    pass


AzureCloudPlatformClient = CloudPlatform

class AzureCloudPlatformIntegration(CloudPlatformIntegrationContract):
    """
    Single public Azure cloud platform entrypoint.

    Legacy catalog factory (create_azure_integration) owns catalog behavior; legacy factories use from_client().
    """

    config: AzureCloudPlatformIntegrationConfig = AzureCloudPlatformIntegrationConfig()
    _client: AzureCloudPlatformClient | None = PrivateAttr(default=None)
    

    @property
    def default_region(self):
        return attribute_access.optional_str(self._require_client(), 'default_region')

    def health(self):
        return self._require_client().health()

    def resolve(self, category):
        return self._require_client().resolve(category)

    @property
    def slug(self):
        return attribute_access.optional_str(self._require_client(), 'slug')

    def _require_client(self) -> CloudPlatform:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


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

CloudPlatform.register(AzureCloudPlatformIntegration)

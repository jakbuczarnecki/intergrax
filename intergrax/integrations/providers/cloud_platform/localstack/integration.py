# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Localstack cloud platform integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.runtime.integrations.categories.devops import CloudPlatformIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

LOCALSTACK_CLOUD_PLATFORM_PROVIDER_ID = "localstack"


class LocalstackCloudPlatformIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Localstack cloud platform integration."""

    pass


LocalstackCloudPlatformClient = CloudPlatform

class LocalstackCloudPlatformIntegration(CloudPlatformIntegrationContract):
    """
    Single public Localstack cloud platform entrypoint.

    Legacy catalog factory (create_localstack_cloud_platform) owns catalog behavior; legacy factories use from_client().
    """

    config: LocalstackCloudPlatformIntegrationConfig = LocalstackCloudPlatformIntegrationConfig()
    _client: LocalstackCloudPlatformClient | None = PrivateAttr(default=None)
    

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

CloudPlatform.register(LocalstackCloudPlatformIntegration)

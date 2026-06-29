# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Gcp cloud platform integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.runtime.integrations.categories.devops import CloudPlatformIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GCP_CLOUD_PLATFORM_PROVIDER_ID = "gcp"


class GcpCloudPlatformIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Gcp cloud platform integration."""

    pass


@runtime_checkable
class GcpCloudPlatformClient(CloudPlatform, Protocol):
    """GCP cloud platform client with project identifier."""

    @property
    def project_id(self) -> str: ...


class GcpCloudPlatformIntegration(CloudPlatformIntegrationContract):
    """
    Single public Gcp cloud platform entrypoint.

    Legacy catalog factory (create_gcp_integration) owns catalog behavior; legacy factories use from_client().
    """

    config: GcpCloudPlatformIntegrationConfig = GcpCloudPlatformIntegrationConfig()
    _client: GcpCloudPlatformClient | None = PrivateAttr(default=None)
    

    @property
    def default_region(self):
        return self._require_client().default_region

    @property
    def project_id(self):
        return self._require_client().project_id

    def health(self):
        return self._require_client().health()

    def resolve(self, category):
        return self._require_client().resolve(category)

    @property
    def slug(self):
        return self._require_client().slug

    def _require_client(self) -> GcpCloudPlatformClient:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: GcpCloudPlatformClient,
        *,
        enabled: bool = False,
    ) -> GcpCloudPlatformIntegration:
        integration = cls.for_provider(
            provider_id=GCP_CLOUD_PLATFORM_PROVIDER_ID,
            display_name="Gcp",
            config=GcpCloudPlatformIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> GcpCloudPlatformClient | None:
        return self._client

CloudPlatform.register(GcpCloudPlatformIntegration)

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Kubernetes cloud platform integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.runtime.integrations.categories.devops import CloudPlatformIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

KUBERNETES_CLOUD_PLATFORM_PROVIDER_ID = "kubernetes"


class KubernetesCloudPlatformIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Kubernetes cloud platform integration."""

    pass


KubernetesCloudPlatformClient = CloudPlatform

class KubernetesCloudPlatformIntegration(CloudPlatformIntegrationContract):
    """
    Single public Kubernetes cloud platform entrypoint.

    Legacy catalog factory (create_kubernetes_cloud_platform) owns catalog behavior; legacy factories use from_client().
    """

    config: KubernetesCloudPlatformIntegrationConfig = KubernetesCloudPlatformIntegrationConfig()
    _client: KubernetesCloudPlatformClient | None = PrivateAttr(default=None)
    

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
        client: KubernetesCloudPlatformClient,
        *,
        enabled: bool = False,
    ) -> KubernetesCloudPlatformIntegration:
        integration = cls.for_provider(
            provider_id=KUBERNETES_CLOUD_PLATFORM_PROVIDER_ID,
            display_name="Kubernetes",
            config=KubernetesCloudPlatformIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> KubernetesCloudPlatformClient | None:
        return self._client

CloudPlatform.register(KubernetesCloudPlatformIntegration)

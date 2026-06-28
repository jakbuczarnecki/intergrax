# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Kubernetes cloud platform integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.devops import CloudPlatformIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

KUBERNETES_CLOUD_PLATFORM_PROVIDER_ID = "kubernetes"


class KubernetesCloudPlatformIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Kubernetes cloud platform integration."""

    pass


@runtime_checkable
class KubernetesCloudPlatformClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class KubernetesCloudPlatformIntegration(CloudPlatformIntegrationContract):
    """
    Kubernetes cloud platform integration.

    The legacy facade (create_kubernetes_cloud_platform) remains separate and backward-compatible.
    """

    config: KubernetesCloudPlatformIntegrationConfig = KubernetesCloudPlatformIntegrationConfig()
    _client: KubernetesCloudPlatformClient | None = PrivateAttr(default=None)

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

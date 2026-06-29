# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Kubernetes cloud platform integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.cloud_platform import CloudPlatform
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
    Single public Kubernetes cloud platform entrypoint.

    Legacy catalog factory (create_kubernetes_cloud_platform) delegates to this class.
    """

    config: KubernetesCloudPlatformIntegrationConfig = KubernetesCloudPlatformIntegrationConfig()
    _client: KubernetesCloudPlatformClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> KubernetesCloudPlatformIntegration:
        integration = cls.for_provider(
            provider_id=KUBERNETES_CLOUD_PLATFORM_PROVIDER_ID,
            display_name="Kubernetes",
            config=KubernetesCloudPlatformIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Kubernetes integration requires a runtime delegate")
        return self._runtime



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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

CloudPlatform.register(KubernetesCloudPlatformIntegration)

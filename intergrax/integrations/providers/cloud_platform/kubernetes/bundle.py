# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_kubernetes_cloud_platform as _legacy_create_kubernetes_cloud_platform

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.cloud_platform.kubernetes.integration import (
    KUBERNETES_CLOUD_PLATFORM_PROVIDER_ID,
    KubernetesCloudPlatformIntegration,
    KubernetesCloudPlatformIntegrationConfig,
    KubernetesCloudPlatformClient,
)

__all__ = [
    "create_kubernetes_cloud_platform",
    "create_kubernetes_cloud_platform_integration",
]


def create_kubernetes_cloud_platform_integration(
    *,
    client: KubernetesCloudPlatformClient | None = None,
    enabled: bool = False,
) -> KubernetesCloudPlatformIntegration:
    """
    Build a contract-based Kubernetes cloud platform integration.

    The legacy facade (create_kubernetes_cloud_platform) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Kubernetes cloud platform integration requires an injected client when enabled=True",
        )
    if client is not None:
        return KubernetesCloudPlatformIntegration.from_client(client, enabled=enabled)
    return KubernetesCloudPlatformIntegration.for_provider(
        provider_id=KUBERNETES_CLOUD_PLATFORM_PROVIDER_ID,
        display_name="Kubernetes",
        config=KubernetesCloudPlatformIntegrationConfig(enabled=enabled),
    )


def create_kubernetes_cloud_platform(**kwargs: object) -> KubernetesCloudPlatformIntegration:
    """Compatibility shim — constructs KubernetesCloudPlatformIntegration from legacy runtime."""
    runtime = _legacy_create_kubernetes_cloud_platform(**kwargs)
    if isinstance(runtime, KubernetesCloudPlatformIntegration):
        return runtime
    return KubernetesCloudPlatformIntegration.from_client(runtime)

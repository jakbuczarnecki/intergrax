# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "KUBERNETES_CLOUD_PLATFORM_PROVIDER_ID",
    "KubernetesCloudPlatformIntegration",
    "KubernetesCloudPlatformIntegrationConfig",
    "KubernetesCloudPlatformClient",
    "create_kubernetes_cloud_platform",
    "create_kubernetes_cloud_platform_integration",
    "register_kubernetes_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_kubernetes_cloud_platform",
        "create_kubernetes_cloud_platform_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "KUBERNETES_CLOUD_PLATFORM_PROVIDER_ID",
        "KubernetesCloudPlatformIntegration",
        "KubernetesCloudPlatformIntegrationConfig",
        "KubernetesCloudPlatformClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "KUBERNETES_CLOUD_PLATFORM_PROVIDER_ID",
        "KubernetesCloudPlatformIntegration",
        "KubernetesCloudPlatformIntegrationConfig",
        "KubernetesCloudPlatformClient",
    }
)

def __getattr__(name: str):
    if name == "register_kubernetes_integration":
        from intergrax.integrations.providers.cloud_platform.kubernetes.register import register_kubernetes_integration

        return register_kubernetes_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.cloud_platform.kubernetes import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.cloud_platform.kubernetes import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.cloud_platform.kubernetes import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

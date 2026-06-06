# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_kubernetes_cloud_platform", "register_kubernetes_integration"]

def __getattr__(name: str):
    if name == "register_kubernetes_integration":
        from intergrax.integrations.providers.cloud_platform.kubernetes.register import register_kubernetes_integration
        return register_kubernetes_integration
    if name == "create_kubernetes_cloud_platform":
        from intergrax.integrations.providers.cloud_platform.kubernetes.bundle import create_kubernetes_cloud_platform
        return create_kubernetes_cloud_platform
    raise AttributeError(name)

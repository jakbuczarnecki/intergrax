# Kubernetes (kubernetes)

Category: `cloud_platform`

## Single public entrypoint

- **`KubernetesCloudPlatformIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `KubernetesCloudPlatformIntegration`.
- Contract factory: `create_kubernetes_cloud_platform_integration()`.

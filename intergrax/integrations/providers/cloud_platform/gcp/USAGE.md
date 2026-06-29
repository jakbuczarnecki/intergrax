# Gcp (gcp)

Category: `cloud_platform`

## Single public entrypoint

- **`GcpCloudPlatformIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `GcpCloudPlatformIntegration`.
- Contract factory: `create_gcp_cloud_platform_integration()`.

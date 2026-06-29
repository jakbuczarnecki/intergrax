# Localstack (localstack)

Category: `cloud_platform`

## Single public entrypoint

- **`LocalstackCloudPlatformIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `LocalstackCloudPlatformIntegration`.
- Contract factory: `create_localstack_cloud_platform_integration()`.

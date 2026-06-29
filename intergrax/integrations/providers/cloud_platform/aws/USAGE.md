# Aws (aws)

Category: `cloud_platform`

## Single public entrypoint

- **`AwsCloudPlatformIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `AwsCloudPlatformIntegration`.
- Contract factory: `create_aws_cloud_platform_integration()`.

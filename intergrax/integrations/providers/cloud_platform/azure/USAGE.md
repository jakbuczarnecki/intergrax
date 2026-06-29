# Azure (azure)

Category: `cloud_platform`

## Single public entrypoint

- **`AzureCloudPlatformIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `AzureCloudPlatformIntegration`.
- Contract factory: `create_azure_cloud_platform_integration()`.

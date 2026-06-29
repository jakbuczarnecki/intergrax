# Azure Blob (azure_blob)

Category: `object_storage`

## Single public entrypoint

- **`AzureBlobObjectStorageIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `AzureBlobObjectStorageIntegration`.
- Contract factory: `create_azure_blob_object_storage_integration()`.

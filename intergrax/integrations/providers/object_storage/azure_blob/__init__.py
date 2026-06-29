# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "AZURE_BLOB_OBJECT_STORAGE_PROVIDER_ID",
    "AzureBlobObjectStorageIntegration",
    "AzureBlobObjectStorageIntegrationConfig",
    "AzureBlobObjectStorageClient",
    "create_azure_blob_object_storage",
    "create_azure_blob_object_storage_integration",
    "register_azure_blob_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_azure_blob_object_storage",
        "create_azure_blob_object_storage_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "AZURE_BLOB_OBJECT_STORAGE_PROVIDER_ID",
        "AzureBlobObjectStorageIntegration",
        "AzureBlobObjectStorageIntegrationConfig",
        "AzureBlobObjectStorageClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "AZURE_BLOB_OBJECT_STORAGE_PROVIDER_ID",
        "AzureBlobObjectStorageIntegration",
        "AzureBlobObjectStorageIntegrationConfig",
        "AzureBlobObjectStorageClient",
    }
)

def __getattr__(name: str):
    if name == "register_azure_blob_integration":
        from intergrax.integrations.providers.object_storage.azure_blob.register import register_azure_blob_integration

        return register_azure_blob_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.object_storage.azure_blob import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.object_storage.azure_blob import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.object_storage.azure_blob import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "GCS_OBJECT_STORAGE_PROVIDER_ID",
    "GcsObjectStorageIntegration",
    "GcsObjectStorageIntegrationConfig",
    "GcsObjectStorageClient",
    "create_gcs_object_storage",
    "create_gcs_object_storage_integration",
    "register_gcs_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_gcs_object_storage",
        "create_gcs_object_storage_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "GCS_OBJECT_STORAGE_PROVIDER_ID",
        "GcsObjectStorageIntegration",
        "GcsObjectStorageIntegrationConfig",
        "GcsObjectStorageClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "GCS_OBJECT_STORAGE_PROVIDER_ID",
        "GcsObjectStorageIntegration",
        "GcsObjectStorageIntegrationConfig",
        "GcsObjectStorageClient",
    }
)

def __getattr__(name: str):
    if name == "register_gcs_integration":
        from intergrax.integrations.providers.object_storage.gcs.register import register_gcs_integration

        return register_gcs_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.object_storage.gcs import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.object_storage.gcs import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.object_storage.gcs import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

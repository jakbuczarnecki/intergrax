# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "MINIO_OBJECT_STORAGE_PROVIDER_ID",
    "MinioObjectStorageIntegration",
    "MinioObjectStorageIntegrationConfig",
    "MinioObjectStorageClient",
    "create_minio_object_storage",
    "create_minio_object_storage_integration",
    "register_minio_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_minio_object_storage",
        "create_minio_object_storage_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "MINIO_OBJECT_STORAGE_PROVIDER_ID",
        "MinioObjectStorageIntegration",
        "MinioObjectStorageIntegrationConfig",
        "MinioObjectStorageClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "MINIO_OBJECT_STORAGE_PROVIDER_ID",
        "MinioObjectStorageIntegration",
        "MinioObjectStorageIntegrationConfig",
        "MinioObjectStorageClient",
    }
)

def __getattr__(name: str):
    if name == "register_minio_integration":
        from intergrax.integrations.providers.object_storage.minio.register import register_minio_integration

        return register_minio_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.object_storage.minio import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.object_storage.minio import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.object_storage.minio import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

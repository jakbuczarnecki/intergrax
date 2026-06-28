# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "FILESYSTEM_OBJECT_STORAGE_PROVIDER_ID",
    "FilesystemObjectStorageIntegration",
    "FilesystemObjectStorageIntegrationConfig",
    "FilesystemObjectStorageClient",
    "create_filesystem_object_storage",
    "create_filesystem_object_storage_integration",
    "register_filesystem_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_filesystem_object_storage",
        "create_filesystem_object_storage_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "FILESYSTEM_OBJECT_STORAGE_PROVIDER_ID",
        "FilesystemObjectStorageIntegration",
        "FilesystemObjectStorageIntegrationConfig",
        "FilesystemObjectStorageClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "FILESYSTEM_OBJECT_STORAGE_PROVIDER_ID",
        "FilesystemObjectStorageIntegration",
        "FilesystemObjectStorageIntegrationConfig",
        "FilesystemObjectStorageClient",
    }
)

def __getattr__(name: str):
    if name == "register_filesystem_integration":
        from intergrax.integrations.providers.object_storage.filesystem.register import register_filesystem_integration

        return register_filesystem_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.object_storage.filesystem import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.object_storage.filesystem import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.object_storage.filesystem import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

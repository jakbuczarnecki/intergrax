# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "GOOGLE_DRIVE_OBJECT_STORAGE_PROVIDER_ID",
    "GoogleDriveObjectStorageIntegration",
    "GoogleDriveObjectStorageIntegrationConfig",
    "GoogleDriveObjectStorageClient",
    "create_google_drive_object_storage",
    "create_google_drive_object_storage_integration",
    "register_google_drive_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_google_drive_object_storage",
        "create_google_drive_object_storage_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "GOOGLE_DRIVE_OBJECT_STORAGE_PROVIDER_ID",
        "GoogleDriveObjectStorageIntegration",
        "GoogleDriveObjectStorageIntegrationConfig",
        "GoogleDriveObjectStorageClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "GOOGLE_DRIVE_OBJECT_STORAGE_PROVIDER_ID",
        "GoogleDriveObjectStorageIntegration",
        "GoogleDriveObjectStorageIntegrationConfig",
        "GoogleDriveObjectStorageClient",
    }
)

def __getattr__(name: str):
    if name == "register_google_drive_integration":
        from intergrax.integrations.providers.object_storage.google_drive.register import register_google_drive_integration

        return register_google_drive_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.object_storage.google_drive import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.object_storage.google_drive import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.object_storage.google_drive import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

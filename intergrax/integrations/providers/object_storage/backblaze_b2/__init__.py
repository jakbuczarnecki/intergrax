# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "BACKBLAZE_B2_OBJECT_STORAGE_PROVIDER_ID",
    "BackblazeB2ObjectStorageIntegration",
    "BackblazeB2ObjectStorageIntegrationConfig",
    "BackblazeB2ObjectStorageClient",
    "create_backblaze_b2_object_storage",
    "create_backblaze_b2_object_storage_integration",
    "register_backblaze_b2_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_backblaze_b2_object_storage",
        "create_backblaze_b2_object_storage_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "BACKBLAZE_B2_OBJECT_STORAGE_PROVIDER_ID",
        "BackblazeB2ObjectStorageIntegration",
        "BackblazeB2ObjectStorageIntegrationConfig",
        "BackblazeB2ObjectStorageClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "BACKBLAZE_B2_OBJECT_STORAGE_PROVIDER_ID",
        "BackblazeB2ObjectStorageIntegration",
        "BackblazeB2ObjectStorageIntegrationConfig",
        "BackblazeB2ObjectStorageClient",
    }
)

def __getattr__(name: str):
    if name == "register_backblaze_b2_integration":
        from intergrax.integrations.providers.object_storage.backblaze_b2.register import register_backblaze_b2_integration

        return register_backblaze_b2_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.object_storage.backblaze_b2 import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.object_storage.backblaze_b2 import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.object_storage.backblaze_b2 import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

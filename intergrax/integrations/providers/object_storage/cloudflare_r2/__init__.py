# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "CLOUDFLARE_R2_OBJECT_STORAGE_PROVIDER_ID",
    "CloudflareR2ObjectStorageIntegration",
    "CloudflareR2ObjectStorageIntegrationConfig",
    "CloudflareR2ObjectStorageClient",
    "create_cloudflare_r2_object_storage",
    "create_cloudflare_r2_object_storage_integration",
    "register_cloudflare_r2_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_cloudflare_r2_object_storage",
        "create_cloudflare_r2_object_storage_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "CLOUDFLARE_R2_OBJECT_STORAGE_PROVIDER_ID",
        "CloudflareR2ObjectStorageIntegration",
        "CloudflareR2ObjectStorageIntegrationConfig",
        "CloudflareR2ObjectStorageClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "CLOUDFLARE_R2_OBJECT_STORAGE_PROVIDER_ID",
        "CloudflareR2ObjectStorageIntegration",
        "CloudflareR2ObjectStorageIntegrationConfig",
        "CloudflareR2ObjectStorageClient",
    }
)

def __getattr__(name: str):
    if name == "register_cloudflare_r2_integration":
        from intergrax.integrations.providers.object_storage.cloudflare_r2.register import register_cloudflare_r2_integration

        return register_cloudflare_r2_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.object_storage.cloudflare_r2 import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.object_storage.cloudflare_r2 import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.object_storage.cloudflare_r2 import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

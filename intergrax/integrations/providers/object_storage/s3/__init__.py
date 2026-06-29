# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""S3 object storage integration (Phase M.6 P2)."""

from intergrax.utils.lazy_export import export_from_bundle
from intergrax.integrations.providers.object_storage.s3.config import (
    ENV_S3_BUCKET,
    ENV_S3_ENDPOINT_URL,
    ENV_S3_PREFIX,
    ENV_S3_REGION,
    S3IntegrationConfig,
)

__all__ = [
    "ENV_S3_BUCKET",
    "ENV_S3_ENDPOINT_URL",
    "ENV_S3_PREFIX",
    "ENV_S3_REGION",
    "S3IntegrationBundle",
    "S3IntegrationConfig",
    "S3ObjectStorage",
    "create_s3_integration",
    "create_s3_object_storage",
    "register_s3_integration",
    "resolve_s3_config",
    "create_s3_object_storage_integration",
]

_LAZY_EXPORTS = frozenset(
    {
        "S3IntegrationBundle",
        "S3ObjectStorage",
        "create_s3_integration",
        "create_s3_object_storage",
        "register_s3_integration",
        "resolve_s3_config",
        "create_s3_object_storage_integration",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "S3_OBJECT_STORAGE_PROVIDER_ID",
        "S3ObjectStorageIntegration",
        "S3ObjectStorageIntegrationConfig",
        "S3ObjectStorageClient",
    }
)

def __getattr__(name: str):
    if name == "register_s3_integration":
        from intergrax.integrations.providers.object_storage.s3.register import register_s3_integration

        return register_s3_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.object_storage.s3 import bundle as _bundle

        return export_from_bundle(_bundle, name, _LAZY_EXPORTS)
    if name == "S3ObjectStorage":
        from intergrax.integrations.providers.object_storage.s3.adapter import _S3ObjectStorage

        return S3ObjectStorage
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.object_storage.s3 import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

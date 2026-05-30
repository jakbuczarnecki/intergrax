# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""S3 object storage integration (Phase M.6 P2)."""

from intergrax.integrations.providers.s3.config import (
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
]

_LAZY_EXPORTS = frozenset(
    {
        "S3IntegrationBundle",
        "S3ObjectStorage",
        "create_s3_integration",
        "create_s3_object_storage",
        "register_s3_integration",
        "resolve_s3_config",
    }
)


def __getattr__(name: str):
    if name == "register_s3_integration":
        from intergrax.integrations.providers.s3.register import register_s3_integration

        return register_s3_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.s3 import bundle as _bundle

        return getattr(_bundle, name)
    if name == "S3ObjectStorage":
        from intergrax.integrations.providers.s3.adapter import S3ObjectStorage

        return S3ObjectStorage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

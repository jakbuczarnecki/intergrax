# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""GCP cloud platform integration (Phase M.6)."""

from intergrax.integrations.providers.cloud_platform.gcp.config import (
    ENV_GCP_CREDENTIALS_FILE,
    ENV_GCP_PROJECT_ID,
    ENV_GCP_REGION,
    GcpIntegrationConfig,
)

__all__ = [
    "ENV_GCP_CREDENTIALS_FILE",
    "ENV_GCP_PROJECT_ID",
    "ENV_GCP_REGION",
    "GcpCloudPlatform",
    "GcpIntegrationBundle",
    "GcpIntegrationConfig",
    "create_gcp_cloud_platform",
    "create_gcp_integration",
    "register_gcp_integration",
    "resolve_gcp_config",
]

_LAZY_EXPORTS = frozenset(
    {
        "GcpIntegrationBundle",
        "GcpCloudPlatform",
        "create_gcp_integration",
        "create_gcp_cloud_platform",
        "register_gcp_integration",
        "resolve_gcp_config",
    }
)


def __getattr__(name: str):
    if name == "register_gcp_integration":
        from intergrax.integrations.providers.cloud_platform.gcp.register import register_gcp_integration

        return register_gcp_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.cloud_platform.gcp import bundle as _bundle

        return getattr(_bundle, name)
    if name == "GcpCloudPlatform":
        from intergrax.integrations.providers.cloud_platform.gcp.adapter import GcpCloudPlatform

        return GcpCloudPlatform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

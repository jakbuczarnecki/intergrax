# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure cloud platform integration (Phase M.6)."""

from intergrax.utils.lazy_export import export_from_bundle
from intergrax.integrations.providers.cloud_platform.azure.config import (
    ENV_AZURE_CLIENT_ID,
    ENV_AZURE_CLIENT_SECRET,
    ENV_AZURE_LOCATION,
    ENV_AZURE_SUBSCRIPTION_ID,
    ENV_AZURE_TENANT_ID,
    AzureIntegrationConfig,
)

__all__ = [
    "ENV_AZURE_CLIENT_ID",
    "ENV_AZURE_CLIENT_SECRET",
    "ENV_AZURE_LOCATION",
    "ENV_AZURE_SUBSCRIPTION_ID",
    "ENV_AZURE_TENANT_ID",
    "AzureCloudPlatform",
    "AzureIntegrationBundle",
    "AzureIntegrationConfig",
    "create_azure_cloud_platform",
    "create_azure_integration",
    "register_azure_integration",
    "resolve_azure_config",
]

_LAZY_EXPORTS = frozenset(
    {
        "AzureIntegrationBundle",
        "AzureCloudPlatform",
        "create_azure_integration",
        "create_azure_cloud_platform",
        "register_azure_integration",
        "resolve_azure_config",
    }
)


def __getattr__(name: str):
    if name == "register_azure_integration":
        from intergrax.integrations.providers.cloud_platform.azure.register import register_azure_integration

        return register_azure_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.cloud_platform.azure import bundle as _bundle

        return export_from_bundle(_bundle, name, _LAZY_EXPORTS)
    if name == "AzureCloudPlatform":
        from intergrax.integrations.providers.cloud_platform.azure.adapter import AzureCloudPlatform

        return AzureCloudPlatform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

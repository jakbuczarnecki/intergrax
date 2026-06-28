# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "DAYTONA_SANDBOX_HOST_PROVIDER_ID",
    "DaytonaSandboxHostIntegration",
    "DaytonaSandboxHostIntegrationConfig",
    "DaytonaSandboxHostClient",
    "create_daytona_sandbox_host",
    "create_daytona_sandbox_host_integration",
    "register_daytona_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_daytona_sandbox_host",
        "create_daytona_sandbox_host_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "DAYTONA_SANDBOX_HOST_PROVIDER_ID",
        "DaytonaSandboxHostIntegration",
        "DaytonaSandboxHostIntegrationConfig",
        "DaytonaSandboxHostClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "DAYTONA_SANDBOX_HOST_PROVIDER_ID",
        "DaytonaSandboxHostIntegration",
        "DaytonaSandboxHostIntegrationConfig",
        "DaytonaSandboxHostClient",
    }
)

def __getattr__(name: str):
    if name == "register_daytona_integration":
        from intergrax.integrations.providers.sandbox_host.daytona.register import register_daytona_integration

        return register_daytona_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.sandbox_host.daytona import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.sandbox_host.daytona import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.sandbox_host.daytona import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

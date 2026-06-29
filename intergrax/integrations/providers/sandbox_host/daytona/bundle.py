# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_daytona_sandbox_host as _legacy_create_daytona_sandbox_host

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.sandbox_host.daytona.integration import (
    DAYTONA_SANDBOX_HOST_PROVIDER_ID,
    DaytonaSandboxHostIntegration,
    DaytonaSandboxHostIntegrationConfig,
    DaytonaSandboxHostClient,
)

__all__ = [
    "create_daytona_sandbox_host",
    "create_daytona_sandbox_host_integration",
]


def create_daytona_sandbox_host_integration(
    *,
    client: DaytonaSandboxHostClient | None = None,
    enabled: bool = False,
) -> DaytonaSandboxHostIntegration:
    """
    Build a contract-based Daytona sandbox host integration.

    The legacy facade (create_daytona_sandbox_host) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Daytona sandbox host integration requires an injected client when enabled=True",
        )
    if client is not None:
        return DaytonaSandboxHostIntegration.from_client(client, enabled=enabled)
    return DaytonaSandboxHostIntegration.for_provider(
        provider_id=DAYTONA_SANDBOX_HOST_PROVIDER_ID,
        display_name="Daytona",
        config=DaytonaSandboxHostIntegrationConfig(enabled=enabled),
    )


def create_daytona_sandbox_host(**kwargs: object) -> DaytonaSandboxHostIntegration:
    """Compatibility shim — constructs DaytonaSandboxHostIntegration from legacy runtime."""
    runtime = _legacy_create_daytona_sandbox_host(**kwargs)
    if isinstance(runtime, DaytonaSandboxHostIntegration):
        return runtime
    return DaytonaSandboxHostIntegration.from_client(runtime)

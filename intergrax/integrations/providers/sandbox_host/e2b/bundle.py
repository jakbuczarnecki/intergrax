# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_e2b_sandbox_host as _legacy_create_e2b_sandbox_host

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.sandbox_host.e2b.integration import (
    E2B_SANDBOX_HOST_PROVIDER_ID,
    E2bSandboxHostIntegration,
    E2bSandboxHostIntegrationConfig,
    E2bSandboxHostClient,
)

__all__ = [
    "create_e2b_sandbox_host",
    "create_e2b_sandbox_host_integration",
]


def create_e2b_sandbox_host_integration(
    *,
    client: E2bSandboxHostClient | None = None,
    enabled: bool = False,
) -> E2bSandboxHostIntegration:
    """
    Build a contract-based E2B sandbox host integration.

    The legacy facade (create_e2b_sandbox_host) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "E2B sandbox host integration requires an injected client when enabled=True",
        )
    if client is not None:
        return E2bSandboxHostIntegration.from_client(client, enabled=enabled)
    return E2bSandboxHostIntegration.for_provider(
        provider_id=E2B_SANDBOX_HOST_PROVIDER_ID,
        display_name="E2B",
        config=E2bSandboxHostIntegrationConfig(enabled=enabled),
    )


def create_e2b_sandbox_host(**kwargs: object) -> E2bSandboxHostIntegration:
    """Compatibility shim — constructs E2bSandboxHostIntegration from legacy runtime."""
    runtime = _legacy_create_e2b_sandbox_host(**kwargs)
    if isinstance(runtime, E2bSandboxHostIntegration):
        return runtime
    return E2bSandboxHostIntegration.from_runtime(runtime)

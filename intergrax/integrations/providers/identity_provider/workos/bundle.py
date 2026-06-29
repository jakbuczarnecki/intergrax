# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_workos_identity_provider as _legacy_create_workos_identity_provider

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.identity_provider.workos.integration import (
    WORKOS_IDENTITY_PROVIDER_PROVIDER_ID,
    WorkosIdentityProviderIntegration,
    WorkosIdentityProviderIntegrationConfig,
    WorkosIdentityProviderClient,
)

__all__ = [
    "create_workos_identity_provider",
    "create_workos_identity_provider_integration",
]


def create_workos_identity_provider_integration(
    *,
    client: WorkosIdentityProviderClient | None = None,
    enabled: bool = False,
) -> WorkosIdentityProviderIntegration:
    """
    Build a contract-based Workos identity provider integration.

    The legacy facade (create_workos_identity_provider) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Workos identity provider integration requires an injected client when enabled=True",
        )
    if client is not None:
        return WorkosIdentityProviderIntegration.from_client(client, enabled=enabled)
    return WorkosIdentityProviderIntegration.for_provider(
        provider_id=WORKOS_IDENTITY_PROVIDER_PROVIDER_ID,
        display_name="Workos",
        config=WorkosIdentityProviderIntegrationConfig(enabled=enabled),
    )


def create_workos_identity_provider(**kwargs: object) -> WorkosIdentityProviderIntegration:
    """Compatibility shim — constructs WorkosIdentityProviderIntegration from legacy runtime."""
    runtime = _legacy_create_workos_identity_provider(**kwargs)
    if isinstance(runtime, WorkosIdentityProviderIntegration):
        return runtime
    return WorkosIdentityProviderIntegration.from_runtime(runtime)

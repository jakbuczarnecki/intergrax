# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_auth0_identity_provider as _legacy_create_auth0_identity_provider

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.identity_provider.auth0.integration import (
    AUTH0_IDENTITY_PROVIDER_PROVIDER_ID,
    Auth0IdentityProviderIntegration,
    Auth0IdentityProviderIntegrationConfig,
    Auth0IdentityProviderClient,
)

__all__ = [
    "create_auth0_identity_provider",
    "create_auth0_identity_provider_integration",
]


def create_auth0_identity_provider_integration(
    *,
    client: Auth0IdentityProviderClient | None = None,
    enabled: bool = False,
) -> Auth0IdentityProviderIntegration:
    """
    Build a contract-based Auth0 identity provider integration.

    The legacy facade (create_auth0_identity_provider) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Auth0 identity provider integration requires an injected client when enabled=True",
        )
    if client is not None:
        return Auth0IdentityProviderIntegration.from_client(client, enabled=enabled)
    return Auth0IdentityProviderIntegration.for_provider(
        provider_id=AUTH0_IDENTITY_PROVIDER_PROVIDER_ID,
        display_name="Auth0",
        config=Auth0IdentityProviderIntegrationConfig(enabled=enabled),
    )


def create_auth0_identity_provider(**kwargs: object) -> Auth0IdentityProviderIntegration:
    """Compatibility shim — constructs Auth0IdentityProviderIntegration from legacy runtime."""
    runtime = _legacy_create_auth0_identity_provider(**kwargs)
    if isinstance(runtime, Auth0IdentityProviderIntegration):
        return runtime
    return Auth0IdentityProviderIntegration.from_runtime(runtime)

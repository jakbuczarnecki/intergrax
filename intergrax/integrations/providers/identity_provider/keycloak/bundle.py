# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_keycloak_identity_provider as _legacy_create_keycloak_identity_provider

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.identity_provider.keycloak.integration import (
    KEYCLOAK_IDENTITY_PROVIDER_PROVIDER_ID,
    KeycloakIdentityProviderIntegration,
    KeycloakIdentityProviderIntegrationConfig,
    KeycloakIdentityProviderClient,
)

__all__ = [
    "create_keycloak_identity_provider",
    "create_keycloak_identity_provider_integration",
]


def create_keycloak_identity_provider_integration(
    *,
    client: KeycloakIdentityProviderClient | None = None,
    enabled: bool = False,
) -> KeycloakIdentityProviderIntegration:
    """
    Build a contract-based Keycloak identity provider integration.

    The legacy facade (create_keycloak_identity_provider) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Keycloak identity provider integration requires an injected client when enabled=True",
        )
    if client is not None:
        return KeycloakIdentityProviderIntegration.from_client(client, enabled=enabled)
    return KeycloakIdentityProviderIntegration.for_provider(
        provider_id=KEYCLOAK_IDENTITY_PROVIDER_PROVIDER_ID,
        display_name="Keycloak",
        config=KeycloakIdentityProviderIntegrationConfig(enabled=enabled),
    )


def create_keycloak_identity_provider(**kwargs: object) -> KeycloakIdentityProviderIntegration:
    """Compatibility shim — constructs KeycloakIdentityProviderIntegration from legacy runtime."""
    runtime = _legacy_create_keycloak_identity_provider(**kwargs)
    if isinstance(runtime, KeycloakIdentityProviderIntegration):
        return runtime
    return KeycloakIdentityProviderIntegration.from_runtime(runtime)

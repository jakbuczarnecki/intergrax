# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p8.factories import create_clerk_identity_provider

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.identity_provider.clerk.integration import (
    CLERK_IDENTITY_PROVIDER_PROVIDER_ID,
    ClerkIdentityProviderIntegration,
    ClerkIdentityProviderIntegrationConfig,
    ClerkIdentityProviderClient,
)

__all__ = [
    "create_clerk_identity_provider",
    "create_clerk_identity_provider_integration",
]


def create_clerk_identity_provider_integration(
    *,
    client: ClerkIdentityProviderClient | None = None,
    enabled: bool = False,
) -> ClerkIdentityProviderIntegration:
    """
    Build a contract-based Clerk identity provider integration.

    The legacy facade (create_clerk_identity_provider) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Clerk identity provider integration requires an injected client when enabled=True",
        )
    if client is not None:
        return ClerkIdentityProviderIntegration.from_client(client, enabled=enabled)
    return ClerkIdentityProviderIntegration.for_provider(
        provider_id=CLERK_IDENTITY_PROVIDER_PROVIDER_ID,
        display_name="Clerk",
        config=ClerkIdentityProviderIntegrationConfig(enabled=enabled),
    )

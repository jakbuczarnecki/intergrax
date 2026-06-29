# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p8.factories import create_okta_identity_provider as _legacy_create_okta_identity_provider

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.identity_provider.okta.integration import (
    OKTA_IDENTITY_PROVIDER_PROVIDER_ID,
    OktaIdentityProviderIntegration,
    OktaIdentityProviderIntegrationConfig,
    OktaIdentityProviderClient,
)

__all__ = [
    "create_okta_identity_provider",
    "create_okta_identity_provider_integration",
]


def create_okta_identity_provider_integration(
    *,
    client: OktaIdentityProviderClient | None = None,
    enabled: bool = False,
) -> OktaIdentityProviderIntegration:
    """
    Build a contract-based Okta identity provider integration.

    The legacy facade (create_okta_identity_provider) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Okta identity provider integration requires an injected client when enabled=True",
        )
    if client is not None:
        return OktaIdentityProviderIntegration.from_client(client, enabled=enabled)
    return OktaIdentityProviderIntegration.for_provider(
        provider_id=OKTA_IDENTITY_PROVIDER_PROVIDER_ID,
        display_name="Okta",
        config=OktaIdentityProviderIntegrationConfig(enabled=enabled),
    )


def create_okta_identity_provider(**kwargs: object) -> OktaIdentityProviderIntegration:
    """Compatibility shim — constructs OktaIdentityProviderIntegration from legacy runtime."""
    runtime = _legacy_create_okta_identity_provider(**kwargs)
    if isinstance(runtime, OktaIdentityProviderIntegration):
        return runtime
    return OktaIdentityProviderIntegration.from_client(runtime)

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Auth0 identity provider integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.identity_provider import IdentityProviderBackend
from intergrax.runtime.integrations.categories.security import IdentityProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

AUTH0_IDENTITY_PROVIDER_PROVIDER_ID = "auth0"


class Auth0IdentityProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Auth0 identity provider integration."""

    pass


Auth0IdentityProviderClient = IdentityProviderBackend

class Auth0IdentityProviderIntegration(IdentityProviderIntegrationContract):
    """
    Single public Auth0 identity provider entrypoint.

    Legacy catalog factory (create_auth0_identity_provider) owns catalog behavior; legacy factories use from_client().
    """

    config: Auth0IdentityProviderIntegrationConfig = Auth0IdentityProviderIntegrationConfig()
    _client: Auth0IdentityProviderClient | None = PrivateAttr(default=None)
    

    def list_tenants(self, limit: int = 50):
        return self._require_client().list_tenants(limit=limit)

    def userinfo(self, token):
        return self._require_client().userinfo(token)

    def verify_token(self, token):
        return self._require_client().verify_token(token)

    def _require_client(self) -> IdentityProviderBackend:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: Auth0IdentityProviderClient,
        *,
        enabled: bool = False,
    ) -> Auth0IdentityProviderIntegration:
        integration = cls.for_provider(
            provider_id=AUTH0_IDENTITY_PROVIDER_PROVIDER_ID,
            display_name="Auth0",
            config=Auth0IdentityProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> Auth0IdentityProviderClient | None:
        return self._client

IdentityProviderBackend.register(Auth0IdentityProviderIntegration)

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Auth0 identity provider integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.security import IdentityProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

AUTH0_IDENTITY_PROVIDER_PROVIDER_ID = "auth0"


class Auth0IdentityProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Auth0 identity provider integration."""

    pass


@runtime_checkable
class Auth0IdentityProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class Auth0IdentityProviderIntegration(IdentityProviderIntegrationContract):
    """
    Auth0 identity provider integration.

    The legacy facade (create_auth0_identity_provider) remains separate and backward-compatible.
    """

    config: Auth0IdentityProviderIntegrationConfig = Auth0IdentityProviderIntegrationConfig()
    _client: Auth0IdentityProviderClient | None = PrivateAttr(default=None)

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

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Keycloak identity provider integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.security import IdentityProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

KEYCLOAK_IDENTITY_PROVIDER_PROVIDER_ID = "keycloak"


class KeycloakIdentityProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Keycloak identity provider integration."""

    pass


@runtime_checkable
class KeycloakIdentityProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class KeycloakIdentityProviderIntegration(IdentityProviderIntegrationContract):
    """
    Keycloak identity provider integration.

    The legacy facade (create_keycloak_identity_provider) remains separate and backward-compatible.
    """

    config: KeycloakIdentityProviderIntegrationConfig = KeycloakIdentityProviderIntegrationConfig()
    _client: KeycloakIdentityProviderClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: KeycloakIdentityProviderClient,
        *,
        enabled: bool = False,
    ) -> KeycloakIdentityProviderIntegration:
        integration = cls.for_provider(
            provider_id=KEYCLOAK_IDENTITY_PROVIDER_PROVIDER_ID,
            display_name="Keycloak",
            config=KeycloakIdentityProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> KeycloakIdentityProviderClient | None:
        return self._client

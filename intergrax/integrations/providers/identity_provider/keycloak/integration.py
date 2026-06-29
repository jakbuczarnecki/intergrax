# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Keycloak identity provider integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.identity_provider import IdentityProviderBackend
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
    Single public Keycloak identity provider entrypoint.

    Legacy catalog factory (create_keycloak_identity_provider) delegates to this class.
    """

    config: KeycloakIdentityProviderIntegrationConfig = KeycloakIdentityProviderIntegrationConfig()
    _client: KeycloakIdentityProviderClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> KeycloakIdentityProviderIntegration:
        integration = cls.for_provider(
            provider_id=KEYCLOAK_IDENTITY_PROVIDER_PROVIDER_ID,
            display_name="Keycloak",
            config=KeycloakIdentityProviderIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Keycloak integration requires a runtime delegate")
        return self._runtime



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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

IdentityProviderBackend.register(KeycloakIdentityProviderIntegration)

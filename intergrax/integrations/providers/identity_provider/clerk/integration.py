# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Clerk identity provider integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.security import IdentityProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

CLERK_IDENTITY_PROVIDER_PROVIDER_ID = "clerk"


class ClerkIdentityProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Clerk identity provider integration."""

    pass


@runtime_checkable
class ClerkIdentityProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class ClerkIdentityProviderIntegration(IdentityProviderIntegrationContract):
    """
    Clerk identity provider integration.

    The legacy facade (create_clerk_identity_provider) remains separate and backward-compatible.
    """

    config: ClerkIdentityProviderIntegrationConfig = ClerkIdentityProviderIntegrationConfig()
    _client: ClerkIdentityProviderClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: ClerkIdentityProviderClient,
        *,
        enabled: bool = False,
    ) -> ClerkIdentityProviderIntegration:
        integration = cls.for_provider(
            provider_id=CLERK_IDENTITY_PROVIDER_PROVIDER_ID,
            display_name="Clerk",
            config=ClerkIdentityProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> ClerkIdentityProviderClient | None:
        return self._client

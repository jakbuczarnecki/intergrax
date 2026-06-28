# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Okta identity provider integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.security import IdentityProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

OKTA_IDENTITY_PROVIDER_PROVIDER_ID = "okta"


class OktaIdentityProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Okta identity provider integration."""

    pass


@runtime_checkable
class OktaIdentityProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class OktaIdentityProviderIntegration(IdentityProviderIntegrationContract):
    """
    Okta identity provider integration.

    The legacy facade (create_okta_identity_provider) remains separate and backward-compatible.
    """

    config: OktaIdentityProviderIntegrationConfig = OktaIdentityProviderIntegrationConfig()
    _client: OktaIdentityProviderClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: OktaIdentityProviderClient,
        *,
        enabled: bool = False,
    ) -> OktaIdentityProviderIntegration:
        integration = cls.for_provider(
            provider_id=OKTA_IDENTITY_PROVIDER_PROVIDER_ID,
            display_name="Okta",
            config=OktaIdentityProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> OktaIdentityProviderClient | None:
        return self._client

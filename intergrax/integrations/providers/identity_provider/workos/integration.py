# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Workos identity provider integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.security import IdentityProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

WORKOS_IDENTITY_PROVIDER_PROVIDER_ID = "workos"


class WorkosIdentityProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Workos identity provider integration."""

    pass


@runtime_checkable
class WorkosIdentityProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class WorkosIdentityProviderIntegration(IdentityProviderIntegrationContract):
    """
    Workos identity provider integration.

    The legacy facade (create_workos_identity_provider) remains separate and backward-compatible.
    """

    config: WorkosIdentityProviderIntegrationConfig = WorkosIdentityProviderIntegrationConfig()
    _client: WorkosIdentityProviderClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: WorkosIdentityProviderClient,
        *,
        enabled: bool = False,
    ) -> WorkosIdentityProviderIntegration:
        integration = cls.for_provider(
            provider_id=WORKOS_IDENTITY_PROVIDER_PROVIDER_ID,
            display_name="Workos",
            config=WorkosIdentityProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> WorkosIdentityProviderClient | None:
        return self._client

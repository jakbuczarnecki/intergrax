# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Mailgun interaction surface integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.collaboration import InteractionSurfaceIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

MAILGUN_INTERACTION_SURFACE_PROVIDER_ID = "mailgun"


class MailgunInteractionSurfaceIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Mailgun interaction surface integration."""

    pass


@runtime_checkable
class MailgunInteractionSurfaceClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class MailgunInteractionSurfaceIntegration(InteractionSurfaceIntegrationContract):
    """
    Mailgun interaction surface integration.

    The legacy facade (create_mailgun_interaction_surface) remains separate and backward-compatible.
    """

    config: MailgunInteractionSurfaceIntegrationConfig = MailgunInteractionSurfaceIntegrationConfig()
    _client: MailgunInteractionSurfaceClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: MailgunInteractionSurfaceClient,
        *,
        enabled: bool = False,
    ) -> MailgunInteractionSurfaceIntegration:
        integration = cls.for_provider(
            provider_id=MAILGUN_INTERACTION_SURFACE_PROVIDER_ID,
            display_name="Mailgun",
            config=MailgunInteractionSurfaceIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> MailgunInteractionSurfaceClient | None:
        return self._client

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slash Command interaction surface integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.collaboration import InteractionSurfaceIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SLASH_COMMAND_INTERACTION_SURFACE_PROVIDER_ID = "slash_command"


class SlashCommandInteractionSurfaceIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Slash Command interaction surface integration."""

    pass


@runtime_checkable
class SlashCommandInteractionSurfaceClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class SlashCommandInteractionSurfaceIntegration(InteractionSurfaceIntegrationContract):
    """
    Slash Command interaction surface integration.

    The legacy facade (create_slash_command_integration) remains separate and backward-compatible.
    """

    config: SlashCommandInteractionSurfaceIntegrationConfig = SlashCommandInteractionSurfaceIntegrationConfig()
    _client: SlashCommandInteractionSurfaceClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: SlashCommandInteractionSurfaceClient,
        *,
        enabled: bool = False,
    ) -> SlashCommandInteractionSurfaceIntegration:
        integration = cls.for_provider(
            provider_id=SLASH_COMMAND_INTERACTION_SURFACE_PROVIDER_ID,
            display_name="Slash Command",
            config=SlashCommandInteractionSurfaceIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> SlashCommandInteractionSurfaceClient | None:
        return self._client

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slash Command interaction surface integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.interaction_surface import InteractionSurface
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
    Single public Slash Command interaction surface entrypoint.

    Legacy catalog factory (create_slash_command_integration) delegates to this class.
    """

    config: SlashCommandInteractionSurfaceIntegrationConfig = SlashCommandInteractionSurfaceIntegrationConfig()
    _client: SlashCommandInteractionSurfaceClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> SlashCommandInteractionSurfaceIntegration:
        integration = cls.for_provider(
            provider_id=SLASH_COMMAND_INTERACTION_SURFACE_PROVIDER_ID,
            display_name="Slash Command",
            config=SlashCommandInteractionSurfaceIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Slash Command integration requires a runtime delegate")
        return self._runtime



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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

InteractionSurface.register(SlashCommandInteractionSurfaceIntegration)

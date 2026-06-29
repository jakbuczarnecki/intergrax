# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slash-command integration bundle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from intergrax.integrations.providers.interaction_surface.slash_command.config import SlashCommandIntegrationConfig
from intergrax.integrations.providers.interaction_surface.slash_command.opens import open_slash_command_interaction_surface
from intergrax.runtime.interactions.adapter_contract import InteractionAdapter


@dataclass(frozen=True)
class SlashCommandIntegrationBundle:
    config: SlashCommandIntegrationConfig
    interaction_surface: InteractionAdapter


def create_slash_command_integration(
    *,
    interaction_adapter: Optional[InteractionAdapter] = None,
    **config_overrides: object,
) -> SlashCommandIntegrationBundle:
    config = SlashCommandIntegrationConfig.from_env(**config_overrides)
    interaction = open_slash_command_interaction_surface(config, implementation=interaction_adapter)
    return SlashCommandIntegrationBundle(config=config, interaction_surface=interaction)


def create_slash_command_interaction_surface(
    *,
    interaction_adapter: Optional[InteractionAdapter] = None,
    **config_overrides: object,
) -> InteractionAdapter:
    """Catalog factory for ``"slash_command"``."""
    return create_slash_command_integration(
        interaction_adapter=interaction_adapter,
        **config_overrides,
    ).interaction_surface

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.interaction_surface.slash_command.integration import (
    SLASH_COMMAND_INTERACTION_SURFACE_PROVIDER_ID,
    SlashCommandInteractionSurfaceIntegration,
    SlashCommandInteractionSurfaceIntegrationConfig,
    SlashCommandInteractionSurfaceClient,
)


def create_slash_command_interaction_surface_integration(
    *,
    client: SlashCommandInteractionSurfaceClient | None = None,
    enabled: bool = False,
) -> SlashCommandInteractionSurfaceIntegration:
    """
    Build a contract-based Slash Command interaction surface integration.

    Compatibility shim — constructs Integration via from_store (create_slash_command_integration) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Slash Command interaction surface integration requires an injected client when enabled=True",
        )
    if client is not None:
        return SlashCommandInteractionSurfaceIntegration.from_client(client, enabled=enabled)
    return SlashCommandInteractionSurfaceIntegration.for_provider(
        provider_id=SLASH_COMMAND_INTERACTION_SURFACE_PROVIDER_ID,
        display_name="Slash Command",
        config=SlashCommandInteractionSurfaceIntegrationConfig(enabled=enabled),
    )

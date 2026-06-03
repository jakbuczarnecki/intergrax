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

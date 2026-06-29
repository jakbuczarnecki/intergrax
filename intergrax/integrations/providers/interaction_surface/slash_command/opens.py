# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slash-command openers."""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.providers.interaction_surface.slash_command.adapter import _SlashCommandIntegrationAdapter
from intergrax.integrations.providers.interaction_surface.slash_command.config import SlashCommandIntegrationConfig
from intergrax.runtime.interactions.adapter_contract import InteractionAdapter


def open_slash_command_interaction_surface(
    config: SlashCommandIntegrationConfig,
    *,
    implementation: Optional[InteractionAdapter] = None,
) -> InteractionAdapter:
    _ = config
    if implementation is not None:
        return implementation
    return _SlashCommandIntegrationAdapter()

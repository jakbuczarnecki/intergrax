# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slash-command interaction adapter — catalog facade."""

from __future__ import annotations

from intergrax.runtime.interactions.adapters.slash_command_adapter import SlashCommandInteractionAdapter


class SlashCommandIntegrationAdapter(SlashCommandInteractionAdapter):
    """Catalog facade over ``SlashCommandInteractionAdapter``."""

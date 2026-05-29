# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slack interaction adapter — wraps ``SlashCommandInteractionAdapter``."""

from __future__ import annotations

from intergrax.runtime.interactions.adapters.slash_command_adapter import SlashCommandInteractionAdapter


class SlackInteractionAdapter(SlashCommandInteractionAdapter):
    """Slack slash-command intake with catalog channel id ``slack``."""

    @property
    def channel(self) -> str:
        return "slack"

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register discord conversation channel in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.conversation_channel.discord.bundle import (
    create_discord_conversation_channel_integration,
)
from intergrax.integrations.providers.conversation_channel.discord.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.conversation_channel.discord.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_discord_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_discord_conversation_channel_integration,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
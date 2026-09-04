# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register discord in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.notification_channel.discord.bundle import create_discord_notification_channel
from intergrax.integrations.providers.notification_channel.discord.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.notification_channel.discord.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_discord_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_discord_notification_channel,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
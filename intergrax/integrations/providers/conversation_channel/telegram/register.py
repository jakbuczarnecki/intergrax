# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register telegram conversation channel in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.conversation_channel.telegram.bundle import (
    create_telegram_conversation_channel_integration,
)
from intergrax.integrations.providers.conversation_channel.telegram.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_telegram_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_telegram_conversation_channel_integration, override=override)

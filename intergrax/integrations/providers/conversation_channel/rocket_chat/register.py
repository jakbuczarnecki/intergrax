# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register rocket_chat conversation channel in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.conversation_channel.rocket_chat.bundle import (
    create_rocket_chat_conversation_channel_integration,
)
from intergrax.integrations.providers.conversation_channel.rocket_chat.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_rocket_chat_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_rocket_chat_conversation_channel_integration, override=override)

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register google_chat conversation channel in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.conversation_channel.google_chat.bundle import (
    create_google_chat_conversation_channel_integration,
)
from intergrax.integrations.providers.conversation_channel.google_chat.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.conversation_channel.google_chat.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_google_chat_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_google_chat_conversation_channel_integration,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
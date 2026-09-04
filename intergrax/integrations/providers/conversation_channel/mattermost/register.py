# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register mattermost conversation channel in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.conversation_channel.mattermost.bundle import (
    create_mattermost_conversation_channel_integration,
)
from intergrax.integrations.providers.conversation_channel.mattermost.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.conversation_channel.mattermost.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_mattermost_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_mattermost_conversation_channel_integration,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
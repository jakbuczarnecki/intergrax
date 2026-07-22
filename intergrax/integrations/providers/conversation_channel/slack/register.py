# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register slack conversation channel in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.conversation_channel.slack.bundle import (
    create_slack_conversation_channel_integration,
)
from intergrax.integrations.providers.conversation_channel.slack.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_slack_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_slack_conversation_channel_integration, override=override)

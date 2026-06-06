# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register mailpit in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.notification_channel.mailpit.bundle import create_mailpit_notification_channel
from intergrax.integrations.providers.notification_channel.mailpit.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_mailpit_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_mailpit_notification_channel, override=override)

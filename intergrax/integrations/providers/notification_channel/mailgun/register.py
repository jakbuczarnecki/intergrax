# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register mailgun in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.notification_channel.mailgun.bundle import (
    create_mailgun_notification_channel,
)
from intergrax.integrations.providers.notification_channel.mailgun.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_mailgun_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_mailgun_notification_channel, override=override)

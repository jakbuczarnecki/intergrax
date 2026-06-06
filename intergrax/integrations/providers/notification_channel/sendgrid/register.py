# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register sendgrid in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.notification_channel.sendgrid.bundle import create_sendgrid_notification_channel
from intergrax.integrations.providers.notification_channel.sendgrid.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_sendgrid_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_sendgrid_notification_channel, override=override)

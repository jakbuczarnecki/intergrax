# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register log in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.notification_channel.log.bundle import create_log_notification_channel
from intergrax.integrations.providers.notification_channel.log.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_log_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_log_notification_channel, override=override)

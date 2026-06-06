# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register incident_io in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.notification_channel.incident_io.bundle import create_incident_io_notification_channel
from intergrax.integrations.providers.notification_channel.incident_io.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_incident_io_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_incident_io_notification_channel, override=override)

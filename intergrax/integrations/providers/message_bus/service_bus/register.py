# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register service_bus in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.message_bus.service_bus.bundle import create_service_bus_message_bus
from intergrax.integrations.providers.message_bus.service_bus.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_service_bus_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_service_bus_message_bus, override=override)

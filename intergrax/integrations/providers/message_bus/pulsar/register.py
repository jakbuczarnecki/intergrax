# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register pulsar in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.message_bus.pulsar.bundle import create_pulsar_message_bus
from intergrax.integrations.providers.message_bus.pulsar.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_pulsar_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_pulsar_message_bus, override=override)

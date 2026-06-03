# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register kafka in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.message_bus.kafka.bundle import create_kafka_message_bus
from intergrax.integrations.providers.message_bus.kafka.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_kafka_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_kafka_message_bus, override=override)

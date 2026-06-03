# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register rabbitmq in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.message_bus.rabbitmq.bundle import create_rabbitmq_message_bus
from intergrax.integrations.providers.message_bus.rabbitmq.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_rabbitmq_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_rabbitmq_message_bus, override=override)

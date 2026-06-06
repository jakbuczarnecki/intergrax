# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register redpanda in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.message_bus.redpanda.bundle import create_redpanda_message_bus
from intergrax.integrations.providers.message_bus.redpanda.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_redpanda_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_redpanda_message_bus, override=override)

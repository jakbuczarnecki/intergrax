# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register sqs in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.message_bus.sqs.bundle import create_sqs_message_bus
from intergrax.integrations.providers.message_bus.sqs.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_sqs_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_sqs_message_bus, override=override)

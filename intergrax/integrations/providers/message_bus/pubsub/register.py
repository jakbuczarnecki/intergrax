# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register pubsub in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.message_bus.pubsub.bundle import create_pubsub_message_bus
from intergrax.integrations.providers.message_bus.pubsub.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.message_bus.pubsub.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_pubsub_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_pubsub_message_bus,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
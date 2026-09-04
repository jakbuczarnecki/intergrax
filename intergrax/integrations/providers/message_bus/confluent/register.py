# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register confluent in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.message_bus.confluent.bundle import create_confluent_message_bus
from intergrax.integrations.providers.message_bus.confluent.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.message_bus.confluent.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_confluent_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_confluent_message_bus,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register nats in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.message_bus.nats.bundle import create_nats_message_bus
from intergrax.integrations.providers.message_bus.nats.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.message_bus.nats.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_nats_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_nats_message_bus,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
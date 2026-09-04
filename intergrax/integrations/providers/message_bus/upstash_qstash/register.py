# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register upstash_qstash in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.message_bus.upstash_qstash.bundle import create_upstash_qstash_message_bus
from intergrax.integrations.providers.message_bus.upstash_qstash.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.message_bus.upstash_qstash.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_upstash_qstash_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_upstash_qstash_message_bus,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
# © Artur Czarnecki. All rights reserved.

"""Reference :class:`IntegrationPlugin` for external packages.

This example implements ``KeyValueCache`` for integration idempotency/rate-limit
patterns — it is **not** wired to Nexus ``TaskMemory`` or ``MemoryView``.
For agent task KV use ``wire_task_memory_from_profile``; for user LTM use
``UserProfileStore`` / sqlite bundle (Phase MEM).
"""

from __future__ import annotations

from intergrax.integrations.contracts.catalog_factory import IntegrationFactoryConfigValue
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.examples.custom_memory_kv.bundle import create_custom_memory_kv_integration
from intergrax.integrations.examples.custom_memory_kv.contract_spec import CONTRACT_SPECS
from intergrax.integrations.examples.custom_memory_kv.integration import CustomMemoryKvIntegration
from intergrax.integrations.examples.custom_memory_kv.manifest import MANIFEST
from intergrax.integrations.registry.contract_spec import IntegrationContractSpec


class CustomMemoryKvPlugin:
    @classmethod
    def integration_manifest(cls) -> IntegrationManifest:
        return MANIFEST

    @classmethod
    def integration_contract_specs(cls) -> tuple[IntegrationContractSpec, ...]:
        return CONTRACT_SPECS

    @classmethod
    def create_integration(cls, **kwargs: IntegrationFactoryConfigValue) -> CustomMemoryKvIntegration:
        _ = kwargs
        return create_custom_memory_kv_integration()

# © Artur Czarnecki. All rights reserved.

"""Entry-point integration plugin for catalog fixture tests."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.contracts.catalog_factory import IntegrationFactoryConfigValue
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.examples.custom_memory_kv.adapter import InProcessKeyValueCache
from intergrax.integrations.examples.custom_memory_kv.integration import CustomMemoryKvIntegration
from intergrax.integrations.registry.contract_spec import IntegrationContractSpec

from intergrax_catalog_fixture.integration_contract import CONTRACT_SPECS


class FixtureKvIntegrationPlugin:
    """Distinct slug from ``custom_memory_kv`` for entry-point-only registration tests."""

    @classmethod
    def integration_manifest(cls) -> IntegrationManifest:
        return IntegrationManifest(
            slug="fixture_ep_kv",
            categories=(IntegrationCategory.KEY_VALUE_CACHE,),
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_FIXTURE_EP_KV",
            description="Fixture entry-point KV plugin for pytest.",
        )

    @classmethod
    def integration_contract_specs(cls) -> tuple[IntegrationContractSpec, ...]:
        return CONTRACT_SPECS

    @classmethod
    def create_integration(cls, **kwargs: IntegrationFactoryConfigValue) -> CustomMemoryKvIntegration:
        _ = kwargs
        return CustomMemoryKvIntegration.from_client(InProcessKeyValueCache())

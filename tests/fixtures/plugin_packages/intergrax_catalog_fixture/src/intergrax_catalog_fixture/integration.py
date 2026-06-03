# © Artur Czarnecki. All rights reserved.

"""Entry-point integration plugin for catalog fixture tests."""

from __future__ import annotations

from typing import Any

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.contracts.key_value_cache import KeyValueCache
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.examples.custom_memory_kv.adapter import InProcessKeyValueCache


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
    def create_integration(cls, **kwargs: Any) -> KeyValueCache:
        _ = kwargs
        return InProcessKeyValueCache()

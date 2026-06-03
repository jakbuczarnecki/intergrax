# © Artur Czarnecki. All rights reserved.

"""Reference :class:`IntegrationPlugin` for external packages."""

from __future__ import annotations

from typing import Any

from intergrax.integrations.contracts.key_value_cache import KeyValueCache
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.examples.custom_memory_kv.adapter import InProcessKeyValueCache
from intergrax.integrations.examples.custom_memory_kv.manifest import MANIFEST


class CustomMemoryKvPlugin:
    @classmethod
    def integration_manifest(cls) -> IntegrationManifest:
        return MANIFEST

    @classmethod
    def create_integration(cls, **kwargs: Any) -> KeyValueCache:
        _ = kwargs
        return InProcessKeyValueCache()

# © Artur Czarnecki. All rights reserved.

"""Reference :class:`IntegrationPlugin` for external packages.

This example implements ``KeyValueCache`` for integration idempotency/rate-limit
patterns — it is **not** wired to Nexus ``TaskMemory`` or ``MemoryView``.
For agent task KV use ``wire_task_memory_from_profile``; for user LTM use
``UserProfileStore`` / sqlite bundle (Phase MEM).
"""

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

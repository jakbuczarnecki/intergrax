# © Artur Czarnecki. All rights reserved.

"""Example external integration — no changes to Intergrax core enums or catalogs."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="custom_memory_kv",
    categories=(IntegrationCategory.KEY_VALUE_CACHE,),
    status=IntegrationStatus.BETA,
    env_prefix="INTERGRAX_CUSTOM_MEMORY_KV",
    description="In-process KV example for third-party integration authors.",
)

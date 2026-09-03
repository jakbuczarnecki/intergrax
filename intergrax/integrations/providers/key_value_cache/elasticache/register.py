# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register elasticache in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.key_value_cache.elasticache.bundle import create_elasticache_key_value_cache
from intergrax.integrations.providers.key_value_cache.elasticache.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.integrations.providers.key_value_cache.elasticache.contract_spec import CONTRACT_SPECS


def register_elasticache_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_elasticache_key_value_cache, override=override, contract_specs=CONTRACT_SPECS)

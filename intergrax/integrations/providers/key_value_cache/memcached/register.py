# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register memcached in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.key_value_cache.memcached.bundle import create_memcached_key_value_cache
from intergrax.integrations.providers.key_value_cache.memcached.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.integrations.providers.key_value_cache.memcached.contract_spec import CONTRACT_SPECS


def register_memcached_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_memcached_key_value_cache, override=override, contract_specs=CONTRACT_SPECS)

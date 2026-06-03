# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.examples.custom_memory_kv import CustomMemoryKvPlugin
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.plugin_register import register_integration_plugin
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.resolve_typed import resolve_key_value_cache

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean() -> None:
    clear_catalog()
    yield
    clear_catalog()


def test_resolve_key_value_cache_typed() -> None:
    register_integration_plugin(CustomMemoryKvPlugin)
    profile = IntegrationProfile(key_value_cache=CustomMemoryKvPlugin)
    cache = resolve_key_value_cache(profile)
    cache.set("t1", "k", b"v")
    assert cache.get("t1", "k") == b"v"


def test_resolve_contract_type_mismatch_raises() -> None:
    register_integration_plugin(CustomMemoryKvPlugin)
    profile = IntegrationProfile(key_value_cache=CustomMemoryKvPlugin)
    with pytest.raises(TypeError, match="expected"):
        from intergrax.integrations.registry.resolve_typed import resolve_contract
        from intergrax.integrations.contracts.relational_store import RelationalStore

        resolve_contract(
            profile,
            IntegrationCategory.KEY_VALUE_CACHE,
            expected=RelationalStore,
        )

# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations._shared.conformance import assert_key_value_cache
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.examples.custom_memory_kv import CustomMemoryKvPlugin, MANIFEST
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.plugin_register import register_integration_plugin
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    yield
    clear_catalog()


def test_external_plugin_registers_and_resolves() -> None:
    register_integration_plugin(CustomMemoryKvPlugin)
    profile = IntegrationProfile(key_value_cache=CustomMemoryKvPlugin)
    cache = profile.resolve(IntegrationCategory.KEY_VALUE_CACHE)
    assert_key_value_cache(cache)
    cache.set("t1", "k", b"v")
    assert cache.get("t1", "k") == b"v"


def test_external_plugin_via_manifest_string_options() -> None:
    register_integration_plugin(CustomMemoryKvPlugin)
    profile = IntegrationProfile(
        key_value_cache=MANIFEST,
        options={MANIFEST.slug: {}},
    )
    cache = profile.resolve(IntegrationCategory.KEY_VALUE_CACHE)
    assert_key_value_cache(cache)

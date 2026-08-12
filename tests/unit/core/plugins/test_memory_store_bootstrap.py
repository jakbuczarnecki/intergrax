# © Artur Czarnecki. All rights reserved.

"""MEM-3.3: bootstrap_memory_stores discovers explicit and entry-point plugins."""

from __future__ import annotations

import pytest

from intergrax.core.memory_bootstrap import bootstrap_memory_stores
from intergrax.core.plugins.discovery import (
    EP_MEMORY_STORES,
    load_entry_point_plugins,
    reset_entry_point_spec_cache_for_tests,
)
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_entry_point_spec_cache() -> None:
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_entry_point_spec_cache_for_tests()


class _ExplicitUserProfileStorePlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "test.explicit_user_profile"

    @classmethod
    def create_user_profile_store(cls, **_kwargs):
        return InMemoryUserProfileStore()


class _ExplicitSessionStoragePlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "test.explicit_session_storage"

    @classmethod
    def create_session_storage(cls, **_kwargs):
        return InMemorySessionStorage()


def test_bootstrap_memory_stores_counts_explicit_plugins() -> None:
    result = bootstrap_memory_stores(
        discover_entry_points=False,
        user_profile_plugins=(_ExplicitUserProfileStorePlugin,),
        session_storage_plugins=(_ExplicitSessionStoragePlugin,),
    )

    assert result.user_profile_plugins == 1
    assert result.session_storage_plugins == 1
    assert result.session_turn_index_plugins == 0


def test_bootstrap_memory_stores_discovers_entry_points_when_enabled() -> None:
    result = bootstrap_memory_stores(discover_entry_points=True)

    assert result.user_profile_plugins >= 0
    assert result.session_storage_plugins >= 0


def test_bootstrap_memory_stores_sums_discovered_and_explicit_plugins() -> None:
    discovered = bootstrap_memory_stores(discover_entry_points=True)
    combined = bootstrap_memory_stores(
        discover_entry_points=True,
        user_profile_plugins=(_ExplicitUserProfileStorePlugin,),
        session_storage_plugins=(_ExplicitSessionStoragePlugin,),
    )

    assert combined.user_profile_plugins == discovered.user_profile_plugins + 1
    assert combined.session_storage_plugins == discovered.session_storage_plugins + 1
    assert combined.session_turn_index_plugins == discovered.session_turn_index_plugins


def test_memory_bootstrap_reuses_cached_entry_point_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.metadata

    from tests.unit.core.plugins.test_plugin_discovery import _EntryPoint, _EntryPoints

    entries = _EntryPoints(
        [
            _EntryPoint(
                "memory",
                "tests.unit.core.plugins.test_plugin_discovery:_DiscoveredPlugin",
                EP_MEMORY_STORES,
            )
        ]
    )
    scan_calls = 0

    def _entry_points() -> _EntryPoints:
        nonlocal scan_calls
        scan_calls += 1
        return entries

    monkeypatch.setattr(importlib.metadata, "entry_points", _entry_points)

    bootstrap_memory_stores(discover_entry_points=True)
    load_entry_point_plugins(EP_MEMORY_STORES)

    assert scan_calls == 1

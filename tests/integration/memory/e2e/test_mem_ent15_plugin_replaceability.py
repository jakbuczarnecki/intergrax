# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-15: PLUGIN CONTRACT E2E — reference vs external plugin provider."""

from __future__ import annotations

import pytest

from intergrax.memory.contracts.memory_control import (
    MemoryControlRecallRequest,
    MemoryControlRememberRequest,
)
from intergrax.memory.resolver.discovery import (
    MemoryStorePluginCatalog,
    discover_classified_memory_store_plugins,
)
from intergrax.memory.resolver.materialization import MemoryStoreMaterializationContext
from intergrax.memory.resolver.resolver import materialize_user_profile_store
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from tests.fixtures.plugin_packages.memory_store_plugin.memory_store_plugin.plugin import (
    ExternalInMemoryUserProfileStorePlugin,
    FixtureExternalUserProfileStore,
)
from tests.integration.memory.e2e.harness import (
    build_in_memory_memory_harness,
    build_plugin_user_profile_harness,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.gate]

_MARKER = "plugin-replaceability-marker"


@pytest.mark.asyncio
async def test_reference_and_plugin_providers_equivalent_semantic_recall() -> None:
    reference = build_in_memory_memory_harness(
        tenant_id="tenant-ent15-ref", user_id="user-plugin"
    )
    identity = reference.identity()
    scope = reference.user_scope(identity)
    await reference.plane.remember(
        identity, scope, MemoryControlRememberRequest(content=_MARKER)
    )
    ref_recall = await reference.plane.recall(
        identity, scope, MemoryControlRecallRequest(query="plugin", top_k=5)
    )

    env = ApplicationEnvironmentProfile.product_defaults()
    discovery = discover_classified_memory_store_plugins(
        discover_entry_points=False,
        explicit_plugins=(ExternalInMemoryUserProfileStorePlugin,),
    )
    catalog = MemoryStorePluginCatalog.from_discovery(discovery)
    ctx = MemoryStoreMaterializationContext(
        tenant_id="tenant-ent15-plugin",
        integration_profile=env.integration_profile,
    )
    plugin_store = materialize_user_profile_store(
        ExternalInMemoryUserProfileStorePlugin.plugin_id(),
        ctx,
        catalog=catalog,
    )
    assert isinstance(plugin_store, FixtureExternalUserProfileStore)

    plugin_harness = build_plugin_user_profile_harness(
        create_store=lambda: plugin_store,
        tenant_id="tenant-ent15-plugin",
        user_id="user-plugin",
    )
    plugin_identity = plugin_harness.identity()
    plugin_scope = plugin_harness.user_scope(plugin_identity)
    await plugin_harness.plane.remember(
        plugin_identity, plugin_scope, MemoryControlRememberRequest(content=_MARKER)
    )
    plugin_recall = await plugin_harness.plane.recall(
        plugin_identity, plugin_scope, MemoryControlRecallRequest(query="plugin", top_k=5)
    )

    assert [item.content for item in ref_recall.items] == [
        item.content for item in plugin_recall.items
    ]

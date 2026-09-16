# © Artur Czarnecki. All rights reserved.

"""Long-horizon memory wiring (MEM-ENT-9)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.memory.contracts.long_horizon_memory import LongHorizonMemoryStore
from intergrax.memory.long_horizon_memory_service import (
    LongHorizonMemoryService,
    build_default_long_horizon_strategies,
)
from intergrax.memory.resolver.discovery import (
    MemoryStorePluginCatalog,
    discover_classified_memory_store_plugins,
)
from intergrax.memory.resolver.materialization import MemoryStoreMaterializationContext
from intergrax.memory.resolver.resolver import materialize_long_horizon_memory_store
from intergrax.memory.stores.in_memory_long_horizon_memory_plugin import (
    DEFAULT_IN_MEMORY_LONG_HORIZON_PLUGIN_ID,
    InMemoryLongHorizonMemoryStorePlugin,
)


def resolve_long_horizon_memory_store(
    env: ApplicationEnvironmentProfile,
) -> LongHorizonMemoryStore | None:
    """Return long-horizon store when memory profile enables the feature."""
    if not env.memory_profile.enable_long_horizon_memory:
        return None

    plugin_id = (
        env.memory_profile.long_horizon_memory_store_plugin_id
        or DEFAULT_IN_MEMORY_LONG_HORIZON_PLUGIN_ID
    )
    discovery = discover_classified_memory_store_plugins(
        discover_entry_points=True,
        explicit_plugins=(InMemoryLongHorizonMemoryStorePlugin,),
    )
    catalog = MemoryStorePluginCatalog.from_discovery(discovery)
    ctx = MemoryStoreMaterializationContext(
        env=env,
        tenant_id=None,
        integration_profile=env.integration_profile,
        rag_stack=None,
    )
    return materialize_long_horizon_memory_store(plugin_id, ctx, catalog=catalog)


def resolve_long_horizon_memory_capability(
    env: ApplicationEnvironmentProfile,
) -> LongHorizonMemoryService | None:
    """Materialize long-horizon memory capability when enabled."""
    store = resolve_long_horizon_memory_store(env)
    if store is None:
        return None
    return LongHorizonMemoryService(
        _store=store,
        _strategies=build_default_long_horizon_strategies(),
    )

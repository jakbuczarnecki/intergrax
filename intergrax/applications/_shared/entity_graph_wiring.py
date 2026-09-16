# © Artur Czarnecki. All rights reserved.

"""Entity graph memory wiring (MEM-ENT-7)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications._shared.memory_security_governance_wiring import (
    resolve_memory_security_governance_service,
)
from intergrax.memory.contracts.entity_temporal_memory import EntityTemporalMemoryStore
from intergrax.memory.entity_graph_memory import EntityGraphMemoryStore
from intergrax.memory.entity_temporal_memory_service import EntityTemporalMemoryService
from intergrax.memory.memory_security_governance_service import MemorySecurityGovernanceService
from intergrax.memory.resolver.discovery import (
    MemoryStorePluginCatalog,
    discover_classified_memory_store_plugins,
)
from intergrax.memory.resolver.materialization import MemoryStoreMaterializationContext
from intergrax.memory.resolver.resolver import materialize_entity_temporal_memory_store
from intergrax.memory.stores.in_memory_entity_temporal_memory_plugin import (
    DEFAULT_IN_MEMORY_ENTITY_TEMPORAL_PLUGIN_ID,
    InMemoryEntityTemporalMemoryStorePlugin,
)


def resolve_entity_temporal_memory_store(
    env: ApplicationEnvironmentProfile,
) -> EntityTemporalMemoryStore | None:
    """Return entity/temporal store when memory profile enables entity graph memory."""
    if not env.memory_profile.enable_entity_graph_memory:
        return None

    plugin_id = (
        env.memory_profile.entity_temporal_memory_store_plugin_id
        or DEFAULT_IN_MEMORY_ENTITY_TEMPORAL_PLUGIN_ID
    )
    discovery = discover_classified_memory_store_plugins(
        discover_entry_points=True,
        explicit_plugins=(InMemoryEntityTemporalMemoryStorePlugin,),
    )
    catalog = MemoryStorePluginCatalog.from_discovery(discovery)
    ctx = MemoryStoreMaterializationContext(
        env=env,
        tenant_id=None,
        integration_profile=env.integration_profile,
        rag_stack=None,
    )
    return materialize_entity_temporal_memory_store(plugin_id, ctx, catalog=catalog)


def resolve_entity_temporal_memory_capability(
    env: ApplicationEnvironmentProfile,
    *,
    security_governance: MemorySecurityGovernanceService | None = None,
) -> EntityTemporalMemoryService | None:
    """Materialize governed entity/temporal read capability when enabled."""
    store = resolve_entity_temporal_memory_store(env)
    if store is None:
        return None
    governance = resolve_memory_security_governance_service(
        security_governance=security_governance,
    )
    return EntityTemporalMemoryService(
        _store=store,
        _security_governance=governance,
    )


def resolve_entity_graph_memory_store(
    env: ApplicationEnvironmentProfile,
) -> EntityGraphMemoryStore | None:
    """Legacy facade for composition roots that still expect ``EntityGraphMemoryStore``."""
    backend = resolve_entity_temporal_memory_store(env)
    if backend is None:
        return None
    return EntityGraphMemoryStore(backend=backend)

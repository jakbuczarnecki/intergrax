# © Artur Czarnecki. All rights reserved.

"""Entity graph memory wiring (MEM-ENT-7)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications._shared.memory_observability_wiring import (
    resolve_memory_diagnostic_emitter,
)
from intergrax.applications._shared.memory_security_governance_wiring import (
    resolve_memory_security_governance_service,
)
from intergrax.memory.contracts.memory_observability import MemoryObservabilitySink
from intergrax.memory.entity_memory_indexing import DefaultEntityMemoryIndexer
from intergrax.memory.memory_diagnostic_emitter import MemoryDiagnosticEmitter
from intergrax.memory.contracts.entity_temporal_memory import (
    EntityTemporalMemoryCapability,
    EntityTemporalMemoryStore,
)
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
        tenant_id=None,
        integration_profile=env.integration_profile,
    )
    return materialize_entity_temporal_memory_store(plugin_id, ctx, catalog=catalog)


def resolve_entity_temporal_memory_capability(
    env: ApplicationEnvironmentProfile,
    *,
    security_governance: MemorySecurityGovernanceService | None = None,
    memory_observability_sink: MemoryObservabilitySink | None = None,
    memory_diagnostic_emitter: MemoryDiagnosticEmitter | None = None,
    store: EntityTemporalMemoryStore | None = None,
) -> EntityTemporalMemoryCapability | None:
    """Materialize governed entity/temporal read capability when enabled."""
    resolved_store = store if store is not None else resolve_entity_temporal_memory_store(env)
    if resolved_store is None:
        return None
    emitter = resolve_memory_diagnostic_emitter(
        sink=memory_observability_sink,
        emitter=memory_diagnostic_emitter,
    )
    governance = resolve_memory_security_governance_service(
        security_governance=security_governance,
        memory_diagnostic_emitter=emitter,
    )
    return EntityTemporalMemoryService(
        _store=resolved_store,
        _security_governance=governance,
    )


def resolve_entity_memory_indexer(
    env: ApplicationEnvironmentProfile,
    *,
    security_governance: MemorySecurityGovernanceService | None = None,
    memory_observability_sink: MemoryObservabilitySink | None = None,
    memory_diagnostic_emitter: MemoryDiagnosticEmitter | None = None,
    store: EntityTemporalMemoryStore | None = None,
) -> DefaultEntityMemoryIndexer | None:
    """Materialize governed entity projection indexer when entity memory is enabled."""
    resolved_store = store if store is not None else resolve_entity_temporal_memory_store(env)
    if resolved_store is None:
        return None
    emitter = resolve_memory_diagnostic_emitter(
        sink=memory_observability_sink,
        emitter=memory_diagnostic_emitter,
    )
    governance = resolve_memory_security_governance_service(
        security_governance=security_governance,
        memory_diagnostic_emitter=emitter,
    )
    return DefaultEntityMemoryIndexer(
        resolved_store,
        security_governance=governance,
        diagnostic_emitter=emitter,
    )


def resolve_entity_graph_memory_store(
    env: ApplicationEnvironmentProfile,
) -> EntityGraphMemoryStore | None:
    """Migration-only legacy facade; production wiring must use ``resolve_entity_temporal_memory_capability``."""
    import warnings

    warnings.warn(
        "resolve_entity_graph_memory_store is migration-only (MEM-ENT-11)",
        DeprecationWarning,
        stacklevel=2,
    )
    backend = resolve_entity_temporal_memory_store(env)
    if backend is None:
        return None
    return EntityGraphMemoryStore(backend=backend)

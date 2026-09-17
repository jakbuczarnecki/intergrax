# © Artur Czarnecki. All rights reserved.

"""Long-horizon memory wiring (MEM-ENT-9)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.memory.contracts.long_horizon_memory import (
    CanonicalMemorySourceAuthority,
    LongHorizonMemoryCapability,
    LongHorizonMemoryStore,
    LongHorizonMemoryViolation,
)
from intergrax.memory.contracts.memory_security_governance import (
    CanonicalMemoryGovernanceSourceAuthority,
)
from intergrax.applications._shared.memory_security_governance_wiring import (
    resolve_memory_security_governance_service,
)
from intergrax.memory.long_horizon_memory_service import (
    LongHorizonMemoryService,
    build_default_long_horizon_strategies,
)
from intergrax.applications._shared.memory_observability_wiring import (
    resolve_memory_diagnostic_emitter,
)
from intergrax.memory.contracts.memory_observability import MemoryObservabilitySink
from intergrax.memory.memory_diagnostic_emitter import MemoryDiagnosticEmitter
from intergrax.memory.memory_security_governance_service import MemorySecurityGovernanceService
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
        tenant_id=None,
        integration_profile=env.integration_profile,
    )
    return materialize_long_horizon_memory_store(plugin_id, ctx, catalog=catalog)


def resolve_long_horizon_memory_capability(
    env: ApplicationEnvironmentProfile,
    *,
    source_authority: CanonicalMemorySourceAuthority | None = None,
    governance_source_authority: CanonicalMemoryGovernanceSourceAuthority | None = None,
    security_governance: MemorySecurityGovernanceService | None = None,
    memory_observability_sink: MemoryObservabilitySink | None = None,
    memory_diagnostic_emitter: MemoryDiagnosticEmitter | None = None,
) -> LongHorizonMemoryCapability | None:
    """Materialize long-horizon memory capability when enabled."""
    store = resolve_long_horizon_memory_store(env)
    if store is None:
        return None
    if source_authority is None:
        raise LongHorizonMemoryViolation(
            "long-horizon memory capability requires CanonicalMemorySourceAuthority"
        )
    if governance_source_authority is None:
        raise LongHorizonMemoryViolation(
            "long-horizon memory capability requires CanonicalMemoryGovernanceSourceAuthority"
        )
    emitter = resolve_memory_diagnostic_emitter(
        sink=memory_observability_sink,
        emitter=memory_diagnostic_emitter,
    )
    governance = resolve_memory_security_governance_service(
        security_governance=security_governance,
        memory_diagnostic_emitter=emitter,
    )
    return LongHorizonMemoryService(
        _store=store,
        _strategies=build_default_long_horizon_strategies(),
        _source_authority=source_authority,
        _governance_source_authority=governance_source_authority,
        _security_governance=governance,
        _diagnostic_emitter=emitter,
    )

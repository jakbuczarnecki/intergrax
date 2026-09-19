# © Artur Czarnecki. All rights reserved.

"""Procedural memory wiring (MEM-ENT-8)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.memory.contracts.memory_security_governance import (
    CanonicalMemoryGovernanceSourceAuthority,
)
from intergrax.memory.contracts.procedural_memory import (
    ProcedureMemoryCapability,
    ProcedureMemoryStore,
    ProcedureMemoryViolation,
)
from intergrax.applications._shared.memory_observability_wiring import (
    resolve_memory_diagnostic_emitter,
)
from intergrax.applications._shared.memory_security_governance_wiring import (
    resolve_memory_security_governance_service,
)
from intergrax.memory.contracts.memory_observability import MemoryObservabilitySink
from intergrax.memory.memory_diagnostic_emitter import MemoryDiagnosticEmitter
from intergrax.memory.memory_security_governance_service import MemorySecurityGovernanceService
from intergrax.memory.procedural_memory_service import (
    ProceduralMemoryService,
    build_default_procedural_memory_strategies,
)
from intergrax.memory.resolver.discovery import (
    MemoryStorePluginCatalog,
    discover_classified_memory_store_plugins,
)
from intergrax.memory.resolver.materialization import MemoryStoreMaterializationContext
from intergrax.memory.resolver.resolver import materialize_procedural_memory_store
from intergrax.memory.stores.in_memory_procedural_memory_plugin import (
    DEFAULT_IN_MEMORY_PROCEDURAL_PLUGIN_ID,
    InMemoryProceduralMemoryStorePlugin,
)


def resolve_procedural_memory_store(
    env: ApplicationEnvironmentProfile,
    *,
    discover_entry_points: bool = True,
    explicit_memory_plugins: Sequence[type] = (),
    catalog: MemoryStorePluginCatalog | None = None,
) -> ProcedureMemoryStore | None:
    """Return procedural store when memory profile enables procedural memory."""
    if not env.memory_profile.enable_procedural_memory:
        return None

    plugin_id = (
        env.memory_profile.procedural_memory_store_plugin_id
        or DEFAULT_IN_MEMORY_PROCEDURAL_PLUGIN_ID
    )
    resolved_catalog = catalog
    if resolved_catalog is None:
        discovery = discover_classified_memory_store_plugins(
            discover_entry_points=discover_entry_points,
            explicit_plugins=_procedural_memory_plugin_candidates(explicit_memory_plugins),
        )
        resolved_catalog = MemoryStorePluginCatalog.from_discovery(discovery)
    ctx = MemoryStoreMaterializationContext(
        tenant_id=None,
        integration_profile=env.integration_profile,
    )
    return materialize_procedural_memory_store(plugin_id, ctx, catalog=resolved_catalog)


def _procedural_memory_plugin_candidates(
    explicit_memory_plugins: Sequence[type],
) -> tuple[type, ...]:
    merged: list[type] = [InMemoryProceduralMemoryStorePlugin]
    for plugin_type in explicit_memory_plugins:
        if plugin_type not in merged:
            merged.append(plugin_type)
    return tuple(merged)


def build_procedural_memory_capability(
    store: ProcedureMemoryStore,
    env: ApplicationEnvironmentProfile,
    *,
    governance_source_authority: CanonicalMemoryGovernanceSourceAuthority,
    security_governance: MemorySecurityGovernanceService | None = None,
    memory_observability_sink: MemoryObservabilitySink | None = None,
    memory_diagnostic_emitter: MemoryDiagnosticEmitter | None = None,
) -> ProcedureMemoryCapability:
    """Construct procedural memory capability from a composition-owned store instance."""
    _ = env
    emitter = resolve_memory_diagnostic_emitter(
        sink=memory_observability_sink,
        emitter=memory_diagnostic_emitter,
    )
    governance = resolve_memory_security_governance_service(
        security_governance=security_governance,
        memory_diagnostic_emitter=emitter,
    )
    return ProceduralMemoryService(
        _store=store,
        _strategies=build_default_procedural_memory_strategies(),
        _security_governance=governance,
        _governance_source_authority=governance_source_authority,
        _diagnostic_emitter=emitter,
    )


def resolve_procedural_memory_capability(
    env: ApplicationEnvironmentProfile,
    *,
    store: ProcedureMemoryStore | None = None,
    discover_entry_points: bool = True,
    explicit_memory_plugins: Sequence[type] = (),
    catalog: MemoryStorePluginCatalog | None = None,
    governance_source_authority: CanonicalMemoryGovernanceSourceAuthority | None = None,
    security_governance: MemorySecurityGovernanceService | None = None,
    memory_observability_sink: MemoryObservabilitySink | None = None,
    memory_diagnostic_emitter: MemoryDiagnosticEmitter | None = None,
) -> ProcedureMemoryCapability | None:
    """Materialize procedural memory capability when enabled."""
    resolved_store = (
        store
        if store is not None
        else resolve_procedural_memory_store(
            env,
            discover_entry_points=discover_entry_points,
            explicit_memory_plugins=explicit_memory_plugins,
            catalog=catalog,
        )
    )
    if resolved_store is None:
        return None
    if governance_source_authority is None:
        raise ProcedureMemoryViolation(
            "procedural memory capability requires CanonicalMemoryGovernanceSourceAuthority"
        )
    return build_procedural_memory_capability(
        resolved_store,
        env,
        governance_source_authority=governance_source_authority,
        security_governance=security_governance,
        memory_observability_sink=memory_observability_sink,
        memory_diagnostic_emitter=memory_diagnostic_emitter,
    )

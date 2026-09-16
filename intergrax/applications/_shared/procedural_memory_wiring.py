# © Artur Czarnecki. All rights reserved.

"""Procedural memory wiring (MEM-ENT-8)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.memory.contracts.memory_security_governance import (
    CanonicalMemoryGovernanceSourceAuthority,
)
from intergrax.memory.contracts.procedural_memory import (
    ProcedureMemoryCapability,
    ProcedureMemoryStore,
    ProcedureMemoryViolation,
)
from intergrax.applications._shared.memory_security_governance_wiring import (
    resolve_memory_security_governance_service,
)
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
) -> ProcedureMemoryStore | None:
    """Return procedural store when memory profile enables procedural memory."""
    if not env.memory_profile.enable_procedural_memory:
        return None

    plugin_id = (
        env.memory_profile.procedural_memory_store_plugin_id
        or DEFAULT_IN_MEMORY_PROCEDURAL_PLUGIN_ID
    )
    discovery = discover_classified_memory_store_plugins(
        discover_entry_points=True,
        explicit_plugins=(InMemoryProceduralMemoryStorePlugin,),
    )
    catalog = MemoryStorePluginCatalog.from_discovery(discovery)
    ctx = MemoryStoreMaterializationContext(
        env=env,
        tenant_id=None,
        integration_profile=env.integration_profile,
        rag_stack=None,
    )
    return materialize_procedural_memory_store(plugin_id, ctx, catalog=catalog)


def resolve_procedural_memory_capability(
    env: ApplicationEnvironmentProfile,
    *,
    governance_source_authority: CanonicalMemoryGovernanceSourceAuthority | None = None,
    security_governance: MemorySecurityGovernanceService | None = None,
) -> ProcedureMemoryCapability | None:
    """Materialize procedural memory capability when enabled."""
    store = resolve_procedural_memory_store(env)
    if store is None:
        return None
    if governance_source_authority is None:
        raise ProcedureMemoryViolation(
            "procedural memory capability requires CanonicalMemoryGovernanceSourceAuthority"
        )
    governance = resolve_memory_security_governance_service(
        security_governance=security_governance,
    )
    return ProceduralMemoryService(
        _store=store,
        _strategies=build_default_procedural_memory_strategies(),
        _security_governance=governance,
        _governance_source_authority=governance_source_authority,
    )

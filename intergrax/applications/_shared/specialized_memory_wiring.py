# © Artur Czarnecki. All rights reserved.

"""Specialized memory composition (entity-adjacent procedural / long-horizon surfaces)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from intergrax.applications._shared.long_horizon_memory_wiring import (
    build_long_horizon_memory_capability,
    resolve_long_horizon_memory_store,
)
from intergrax.applications._shared.procedural_memory_wiring import (
    build_procedural_memory_capability,
    resolve_procedural_memory_store,
)
from intergrax.memory.resolver.discovery import MemoryStorePluginCatalog
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.memory.contracts.long_horizon_memory import (
    CanonicalMemorySourceAuthority,
    LongHorizonMemoryCapability,
    LongHorizonMemoryStore,
)
from intergrax.memory.contracts.memory_observability import MemoryObservabilitySink
from intergrax.memory.contracts.memory_security_governance import (
    CanonicalMemoryGovernanceSourceAuthority,
)
from intergrax.memory.contracts.procedural_memory import (
    ProcedureMemoryCapability,
    ProcedureMemoryStore,
)
from intergrax.memory.memory_diagnostic_emitter import MemoryDiagnosticEmitter
from intergrax.memory.memory_security_governance_service import MemorySecurityGovernanceService


@dataclass(frozen=True, slots=True)
class SpecializedMemoryCapabilities:
    """Bounded Tier-3 carrier for procedural and long-horizon memory surfaces."""

    procedural_memory_store: ProcedureMemoryStore | None = None
    procedural_memory_capability: ProcedureMemoryCapability | None = None
    long_horizon_memory_store: LongHorizonMemoryStore | None = None
    long_horizon_memory_capability: LongHorizonMemoryCapability | None = None


def resolve_specialized_memory_capabilities(
    env: ApplicationEnvironmentProfile,
    *,
    discover_entry_points: bool = True,
    explicit_memory_plugins: Sequence[type] = (),
    memory_store_plugin_catalog: MemoryStorePluginCatalog | None = None,
    security_governance: MemorySecurityGovernanceService | None = None,
    memory_observability_sink: MemoryObservabilitySink | None = None,
    memory_diagnostic_emitter: MemoryDiagnosticEmitter | None = None,
    governance_source_authority: CanonicalMemoryGovernanceSourceAuthority | None = None,
    long_horizon_source_authority: CanonicalMemorySourceAuthority | None = None,
) -> SpecializedMemoryCapabilities:
    """
    Resolve specialized stores from ``MemoryProfile`` flags and optional governed capabilities.

    Store materialization is deterministic from feature flags. Capabilities require
    composition-owned canonical authorities (not minted here).
    """
    procedural_store = resolve_procedural_memory_store(
        env,
        discover_entry_points=discover_entry_points,
        explicit_memory_plugins=explicit_memory_plugins,
        catalog=memory_store_plugin_catalog,
    )
    long_horizon_store = resolve_long_horizon_memory_store(
        env,
        discover_entry_points=discover_entry_points,
        explicit_memory_plugins=explicit_memory_plugins,
        catalog=memory_store_plugin_catalog,
    )

    procedural_capability: ProcedureMemoryCapability | None = None
    if procedural_store is not None and governance_source_authority is not None:
        procedural_capability = build_procedural_memory_capability(
            procedural_store,
            env,
            governance_source_authority=governance_source_authority,
            security_governance=security_governance,
            memory_observability_sink=memory_observability_sink,
            memory_diagnostic_emitter=memory_diagnostic_emitter,
        )

    long_horizon_capability: LongHorizonMemoryCapability | None = None
    if (
        long_horizon_store is not None
        and governance_source_authority is not None
        and long_horizon_source_authority is not None
    ):
        long_horizon_capability = build_long_horizon_memory_capability(
            long_horizon_store,
            env,
            source_authority=long_horizon_source_authority,
            governance_source_authority=governance_source_authority,
            security_governance=security_governance,
            memory_observability_sink=memory_observability_sink,
            memory_diagnostic_emitter=memory_diagnostic_emitter,
        )

    return SpecializedMemoryCapabilities(
        procedural_memory_store=procedural_store,
        procedural_memory_capability=procedural_capability,
        long_horizon_memory_store=long_horizon_store,
        long_horizon_memory_capability=long_horizon_capability,
    )

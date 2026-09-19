# © Artur Czarnecki. All rights reserved.

"""Specialized memory composition (entity-adjacent procedural / long-horizon surfaces)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.long_horizon_memory_wiring import (
    resolve_long_horizon_memory_capability,
    resolve_long_horizon_memory_store,
)
from intergrax.applications._shared.procedural_memory_wiring import (
    resolve_procedural_memory_capability,
    resolve_procedural_memory_store,
)
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
    procedural_store = resolve_procedural_memory_store(env)
    long_horizon_store = resolve_long_horizon_memory_store(env)

    procedural_capability: ProcedureMemoryCapability | None = None
    if procedural_store is not None and governance_source_authority is not None:
        procedural_capability = resolve_procedural_memory_capability(
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
        long_horizon_capability = resolve_long_horizon_memory_capability(
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

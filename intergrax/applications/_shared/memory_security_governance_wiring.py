# © Artur Czarnecki. All rights reserved.

"""Shared memory security governance composition (MEM-ENT-10B)."""

from __future__ import annotations

from intergrax.memory.contracts.memory_observability import MemoryObservabilitySink
from intergrax.memory.memory_diagnostic_emitter import MemoryDiagnosticEmitter
from intergrax.memory.memory_security_governance_service import (
    MemorySecurityGovernanceService,
    build_default_memory_security_governance_service,
)
from intergrax.applications._shared.memory_observability_wiring import (
    resolve_memory_diagnostic_emitter,
)

__all__ = ["resolve_memory_security_governance_service"]


def resolve_memory_security_governance_service(
    *,
    security_governance: MemorySecurityGovernanceService | None = None,
    memory_observability_sink: MemoryObservabilitySink | None = None,
    memory_diagnostic_emitter: MemoryDiagnosticEmitter | None = None,
) -> MemorySecurityGovernanceService:
    """Return one governance boundary instance for memory composition roots."""
    if security_governance is not None:
        return security_governance
    emitter = resolve_memory_diagnostic_emitter(
        sink=memory_observability_sink,
        emitter=memory_diagnostic_emitter,
    )
    return build_default_memory_security_governance_service(
        diagnostic_emitter=emitter,
    )

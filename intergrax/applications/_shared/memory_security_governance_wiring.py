# © Artur Czarnecki. All rights reserved.

"""Shared memory security governance composition (MEM-ENT-10B)."""

from __future__ import annotations

from intergrax.memory.memory_security_governance_service import (
    MemorySecurityGovernanceService,
    build_default_memory_security_governance_service,
)

__all__ = ["resolve_memory_security_governance_service"]


def resolve_memory_security_governance_service(
    *,
    security_governance: MemorySecurityGovernanceService | None = None,
) -> MemorySecurityGovernanceService:
    """Return one governance boundary instance for memory composition roots."""
    return security_governance or build_default_memory_security_governance_service()

# © Artur Czarnecki. All rights reserved.

"""Default behavioral qualification checks (MEM-ENT-13)."""

from __future__ import annotations

from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderQualificationCheck,
)
from intergrax.memory.provider_qualification.checks.entity_temporal_memory_store import (
    ENTITY_TEMPORAL_MEMORY_STORE_CHECKS,
)
from intergrax.memory.provider_qualification.checks.long_horizon_memory_store import (
    LONG_HORIZON_MEMORY_STORE_CHECKS,
)
from intergrax.memory.provider_qualification.checks.procedure_memory_store import (
    PROCEDURE_MEMORY_STORE_CHECKS,
)
from intergrax.memory.provider_qualification.checks.user_profile_store import (
    USER_PROFILE_STORE_CHECKS,
)

__all__ = [
    "default_checks_for_capability",
    "ENTITY_TEMPORAL_MEMORY_STORE_CHECKS",
    "LONG_HORIZON_MEMORY_STORE_CHECKS",
    "PROCEDURE_MEMORY_STORE_CHECKS",
    "USER_PROFILE_STORE_CHECKS",
]


def default_checks_for_capability(
    capability: MemoryProviderCapabilityKind,
) -> tuple[MemoryProviderQualificationCheck, ...]:
    if capability is MemoryProviderCapabilityKind.USER_PROFILE_STORE:
        return USER_PROFILE_STORE_CHECKS
    if capability is MemoryProviderCapabilityKind.ENTITY_TEMPORAL_MEMORY_STORE:
        return ENTITY_TEMPORAL_MEMORY_STORE_CHECKS
    if capability is MemoryProviderCapabilityKind.PROCEDURE_MEMORY_STORE:
        return PROCEDURE_MEMORY_STORE_CHECKS
    if capability is MemoryProviderCapabilityKind.LONG_HORIZON_MEMORY_STORE:
        return LONG_HORIZON_MEMORY_STORE_CHECKS
    return ()

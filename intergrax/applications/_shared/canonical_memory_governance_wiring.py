# © Artur Czarnecki. All rights reserved.

"""Canonical memory governance source wiring (MEM-ENT-10B-R)."""

from __future__ import annotations

from intergrax.memory.canonical_memory_governance_source_authority import (
    ReaderBackedCanonicalMemoryGovernanceSourceAuthority,
)
from intergrax.memory.contracts.entity_temporal_memory import EntityMemoryScope
from intergrax.memory.contracts.memory_security_governance import (
    CanonicalMemoryGovernanceEntryReader,
    CanonicalMemoryGovernanceSourceAuthority,
)


def resolve_canonical_memory_governance_source_authority(
    reader: CanonicalMemoryGovernanceEntryReader,
    *,
    bound_scope: EntityMemoryScope | None = None,
) -> CanonicalMemoryGovernanceSourceAuthority:
    """Build governance source authority from a neutral sync entry reader."""
    return ReaderBackedCanonicalMemoryGovernanceSourceAuthority(
        _reader=reader,
        _bound_scope=bound_scope,
    )

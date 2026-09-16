# © Artur Czarnecki. All rights reserved.

"""Canonical memory governance source resolution (MEM-ENT-10B-R)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.memory.contracts.entity_temporal_memory import EntityMemoryScope
from intergrax.memory.contracts.memory_security_governance import (
    CanonicalMemoryGovernanceEntryReader,
    CanonicalMemoryGovernanceSourceAuthority,
    CanonicalMemoryGovernanceSourceNotFound,
    CanonicalMemoryGovernanceSourceSnapshot,
    validate_canonical_governance_source_snapshot,
)

__all__ = ["ReaderBackedCanonicalMemoryGovernanceSourceAuthority"]


def _scope_matches(requested: EntityMemoryScope, bound: EntityMemoryScope) -> bool:
    return (
        requested.tenant_id == bound.tenant_id
        and requested.user_id == bound.user_id
        and requested.workspace_id == bound.workspace_id
    )


@dataclass(frozen=True, slots=True)
class ReaderBackedCanonicalMemoryGovernanceSourceAuthority:
    """``CanonicalMemoryGovernanceSourceAuthority`` backed by a neutral entry reader."""

    _reader: CanonicalMemoryGovernanceEntryReader
    _bound_scope: EntityMemoryScope | None = None

    def resolve_canonical_governance_source(
        self,
        scope: EntityMemoryScope,
        memory_id: str,
        revision: int,
    ) -> CanonicalMemoryGovernanceSourceSnapshot:
        if self._bound_scope is not None and not _scope_matches(scope, self._bound_scope):
            raise CanonicalMemoryGovernanceSourceNotFound("canonical governance scope mismatch")
        entry = self._reader.read_governance_entry(scope, memory_id, revision)
        if entry is None:
            raise CanonicalMemoryGovernanceSourceNotFound(
                f"canonical governance entry not found: {memory_id}@{revision}"
            )
        snapshot = CanonicalMemoryGovernanceSourceSnapshot(
            memory_id=entry.entry_id,
            revision=max(1, int(entry.revision or 1)),
            provenance=entry.provenance,
            trust=entry.trust,
            governance=entry.governance,
            kind=entry.kind,
        )
        validate_canonical_governance_source_snapshot(memory_id, revision, snapshot)
        return snapshot

# © Artur Czarnecki. All rights reserved.

"""Entity graph indexing from LTM entries (MEM-ENT-7 typed projection)."""

from __future__ import annotations

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.entity_temporal_memory import (
    EntityMemoryIndexer,
    EntityMemoryScope,
    EntityTemporalMemoryStore,
)
from intergrax.memory.entity_memory_indexing import DefaultEntityMemoryIndexer
from intergrax.memory.memory_diagnostic_emitter import MemoryDiagnosticEmitter
from intergrax.memory.memory_security_governance_service import MemorySecurityGovernanceService
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry


class EntityGraphMemoryService:
    """Indexes user memory entries via ``EntityMemoryIndexer`` (derived projection)."""

    def __init__(
        self,
        store: EntityTemporalMemoryStore,
        *,
        security_governance: MemorySecurityGovernanceService,
        indexer: EntityMemoryIndexer | None = None,
        diagnostic_emitter: MemoryDiagnosticEmitter | None = None,
    ) -> None:
        self._store = store
        self._security_governance = security_governance
        self._indexer = indexer or DefaultEntityMemoryIndexer(
            store,
            security_governance=security_governance,
            diagnostic_emitter=diagnostic_emitter,
        )

    @property
    def store(self) -> EntityTemporalMemoryStore:
        return self._store

    @property
    def indexer(self) -> EntityMemoryIndexer:
        return self._indexer

    def index_memory_entry(
        self,
        identity: RequestIdentity,
        *,
        tenant_id: str,
        user_id: str,
        entry: UserProfileMemoryEntry,
    ) -> None:
        scope = EntityMemoryScope(tenant_id=tenant_id, user_id=user_id)
        self._indexer.index_memory_entry(identity, scope, entry)

    def remove_memory_entry(
        self,
        identity: RequestIdentity,
        *,
        tenant_id: str,
        user_id: str,
        memory_entry_id: str,
    ) -> None:
        scope = EntityMemoryScope(tenant_id=tenant_id, user_id=user_id)
        self._indexer.remove_memory_entry(identity, scope, memory_entry_id)

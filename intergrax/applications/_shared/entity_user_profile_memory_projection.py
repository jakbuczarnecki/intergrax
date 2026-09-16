# © Artur Czarnecki. All rights reserved.

"""Production UserProfile → Entity memory projection adapter (MEM-ENT-15-R2)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from intergrax.memory.contracts.entity_temporal_memory import (
    EntityMemoryIndexer,
    EntityMemoryScope,
    EntityTemporalMemoryCapability,
    entity_memory_entity_id_for_entry,
)
from intergrax.memory.contracts.memory_lifecycle import (
    MemoryProjectionReconciliationDisposition,
    MemoryProjectionReconciliationResult,
    UserProfileMemoryProjectionContext,
    UserProfileMemoryReconciliationContext,
)
from intergrax.memory.memory_temporal import filter_active_memory_entries
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry

__all__ = ["EntityIndexerUserProfileMemoryProjection"]


@dataclass(slots=True)
class EntityIndexerUserProfileMemoryProjection:
    """Bridges ``UserProfileMemoryProjection`` to a configured ``EntityMemoryIndexer``."""

    indexer: EntityMemoryIndexer
    entity_capability: EntityTemporalMemoryCapability
    workspace_id: str | None = None
    projection_id: str = "entity-temporal"

    def _scope(self, context: UserProfileMemoryProjectionContext) -> EntityMemoryScope:
        tenant = (context.identity.tenant_id or "").strip()
        if not tenant:
            raise ValueError("entity projection requires identity.tenant_id")
        return EntityMemoryScope(
            tenant_id=tenant,
            user_id=context.user_id,
            workspace_id=self.workspace_id,
        )

    async def upsert_memory_entry(
        self,
        context: UserProfileMemoryProjectionContext,
        entry: UserProfileMemoryEntry,
    ) -> None:
        self.indexer.index_memory_entry(
            context.identity,
            self._scope(context),
            entry,
        )

    async def delete_memory_entries(
        self,
        context: UserProfileMemoryProjectionContext,
        entry_ids: Sequence[str],
    ) -> None:
        scope = self._scope(context)
        identity = context.identity
        for memory_entry_id in entry_ids:
            self.indexer.remove_memory_entry(identity, scope, memory_entry_id)

    async def reconcile(
        self,
        context: UserProfileMemoryReconciliationContext,
    ) -> MemoryProjectionReconciliationResult:
        if context.profile is None:
            return MemoryProjectionReconciliationResult(
                projection_id=self.projection_id,
                disposition=MemoryProjectionReconciliationDisposition.CONSISTENT,
            )
        identity = context.identity
        scope = EntityMemoryScope(
            tenant_id=(identity.tenant_id or "").strip(),
            user_id=context.user_id,
            workspace_id=self.workspace_id,
        )
        if not scope.tenant_id:
            raise ValueError("entity projection reconcile requires identity.tenant_id")
        changed = False
        for entry in filter_active_memory_entries(context.profile.memory_entries):
            if entry.entry_id not in context.authoritative_active_entry_ids:
                continue
            entity_id = entity_memory_entity_id_for_entry(scope, entry.entry_id)
            existing = self.entity_capability.get_entity(identity, scope, entity_id)
            if existing is None or existing.source_memory_revision != entry.revision:
                self.indexer.index_memory_entry(identity, scope, entry)
                changed = True
        disposition = (
            MemoryProjectionReconciliationDisposition.REPAIRED
            if changed
            else MemoryProjectionReconciliationDisposition.CONSISTENT
        )
        return MemoryProjectionReconciliationResult(
            projection_id=self.projection_id,
            disposition=disposition,
        )

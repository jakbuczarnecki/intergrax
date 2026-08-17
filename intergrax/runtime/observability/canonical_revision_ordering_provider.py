# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical first-party ``RevisionOrderingAuthority`` provider (TRACE-BITEMP-2)."""

from __future__ import annotations

from intergrax.contracts.bitemporal_knowledge import (
    KnowledgeOrderingScope,
    KnowledgeRevisionAcceptance,
    KnowledgeRevisionId,
    KnowledgeRevisionPosition,
    KnowledgeRevisionPositionLifecycle,
    KnowledgeRevisionPositionRecord,
    KnowledgeRevisionResolutionRecord,
    KnowledgeRevisionResolutionReason,
    KnowledgeRevisionResolutionSource,
    KnowledgeRevisionWatermark,
    OrphanedDurableRevisionRecord,
    ResolutionAuthority,
    RevisionAcceptanceKey,
    RevisionOrderingAuthority,
)
from intergrax.runtime.observability.revision_ordering_store import (
    RevisionOrderingSQLiteStore,
    RevisionOrderingStoreTestHooks,
)


class CanonicalRevisionOrderingProvider(RevisionOrderingAuthority):
    """Tenant-scoped transactional revision ordering over durable SQLite."""

    def __init__(self, store: RevisionOrderingSQLiteStore) -> None:
        self._store = store

    @classmethod
    def open(
        cls,
        db_path: str,
        *,
        test_hooks: RevisionOrderingStoreTestHooks | None = None,
    ) -> CanonicalRevisionOrderingProvider:
        return cls(RevisionOrderingSQLiteStore(db_path, test_hooks=test_hooks))

    def close(self) -> None:
        self._store.close()

    @property
    def store(self) -> RevisionOrderingSQLiteStore:
        return self._store

    def accept_revision(
        self,
        *,
        scope: KnowledgeOrderingScope,
        revision_id: KnowledgeRevisionId,
        acceptance_key: RevisionAcceptanceKey,
    ) -> KnowledgeRevisionAcceptance:
        return self._store.accept_revision(
            scope=scope,
            revision_id=revision_id,
            acceptance_key=acceptance_key,
        )

    def position_lifecycle(
        self,
        position: KnowledgeRevisionPosition,
    ) -> KnowledgeRevisionPositionLifecycle:
        return self._store.position_lifecycle(position)

    def watermark(self, scope: KnowledgeOrderingScope) -> KnowledgeRevisionWatermark:
        return self._store.watermark(scope)

    def records_through(
        self,
        watermark: KnowledgeRevisionWatermark,
    ) -> tuple[KnowledgeRevisionPositionRecord, ...]:
        return self._store.records_through(watermark)

    def unresolved_positions(
        self,
        scope: KnowledgeOrderingScope,
    ) -> tuple[KnowledgeRevisionPosition, ...]:
        return self._store.unresolved_positions(scope)

    def acquire_resolution_authority(
        self,
        scope: KnowledgeOrderingScope,
    ) -> ResolutionAuthority:
        return self._store.acquire_resolution_authority(scope)

    def resolve_unresolved_position(
        self,
        *,
        position: KnowledgeRevisionPosition,
        authority: ResolutionAuthority,
        reason: KnowledgeRevisionResolutionReason,
        source: KnowledgeRevisionResolutionSource,
        actor_identity: str | None = None,
        correlation_id: str | None = None,
    ) -> KnowledgeRevisionResolutionRecord:
        return self._store.resolve_unresolved_position(
            position=position,
            authority=authority,
            reason=reason,
            source=source,
            actor_identity=actor_identity,
            correlation_id=correlation_id,
        )

    def inject_late_physical_write(
        self,
        *,
        position: KnowledgeRevisionPosition,
        revision_id: KnowledgeRevisionId,
        stale_fencing_generation: ResolutionAuthority | None = None,
    ) -> OrphanedDurableRevisionRecord | None:
        from intergrax.contracts.bitemporal_knowledge import RevisionFencingGeneration

        if stale_fencing_generation is None:
            fence = RevisionFencingGeneration(scope=position.scope, value=0)
        else:
            fence = stale_fencing_generation.fencing_generation
        return self._store.inject_late_physical_write(
            position=position,
            revision_id=revision_id,
            stale_fencing_generation=fence,
        )

    def list_orphan_records(
        self,
        scope: KnowledgeOrderingScope,
    ) -> tuple[OrphanedDurableRevisionRecord, ...]:
        return self._store.list_orphan_records(scope)

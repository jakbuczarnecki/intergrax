# © Artur Czarnecki. All rights reserved.

"""Controlled interleaving hooks for in-memory memory stores (test-only)."""

from __future__ import annotations

import threading
from collections.abc import Callable

from intergrax.memory.contracts.entity_temporal_memory import (
    EntityMemoryScope,
    EntityRecord,
)
from intergrax.memory.stores.in_memory_entity_temporal_memory_store import (
    InMemoryEntityTemporalMemoryStore,
)
from intergrax.memory.stores.in_memory_procedural_memory_store import (
    InMemoryProceduralMemoryStore,
)
from intergrax.memory.contracts.procedural_memory import (
    ProceduralMemoryScope,
    ProcedureRecord,
)


class InterleavingEntityTemporalMemoryStore(InMemoryEntityTemporalMemoryStore):
    """Invokes ``pre_commit`` immediately before persisting an entity upsert."""

    pre_commit: Callable[[], None] | None = None

    def upsert_entity(self, scope: EntityMemoryScope, record: EntityRecord) -> EntityRecord:
        hook = self.pre_commit
        if hook is not None:
            self.pre_commit = None
            try:
                hook()
            finally:
                self.pre_commit = hook
        return super().upsert_entity(scope, record)


class InterleavingProceduralMemoryStore(InMemoryProceduralMemoryStore):
    """Invokes ``pre_commit`` immediately before persisting a procedure upsert."""

    pre_commit: Callable[[], None] | None = None

    def upsert_procedure(
        self,
        scope: ProceduralMemoryScope,
        record: ProcedureRecord,
    ) -> ProcedureRecord:
        hook = self.pre_commit
        if hook is not None:
            self.pre_commit = None
            try:
                hook()
            finally:
                self.pre_commit = hook
        return super().upsert_procedure(scope, record)


class OverlapBarrierEntityTemporalMemoryStore(InMemoryEntityTemporalMemoryStore):
    """Synchronizes two revision writers inside ``upsert_entity`` before store lock."""

    overlap_upsert_barrier: threading.Barrier | None = None

    def upsert_entity(self, scope: EntityMemoryScope, record: EntityRecord) -> EntityRecord:
        barrier = self.overlap_upsert_barrier
        incoming_revision = record.source_memory_revision
        if barrier is not None and incoming_revision in {4, 6}:
            barrier.wait(timeout=5.0)
        return super().upsert_entity(scope, record)

# © Artur Czarnecki. All rights reserved.

"""Entity graph indexing from LTM entries (MEM-DEPTH-5.1 runtime integration)."""

from __future__ import annotations

from intergrax.memory.entity_graph_memory import EntityEdge, EntityGraphMemoryStore, EntityNode
from intergrax.memory.user_profile_memory import MemoryKind, UserProfileMemoryEntry


class EntityGraphMemoryService:
    """Indexes user memory entries into the in-process entity graph store."""

    def __init__(self, store: EntityGraphMemoryStore) -> None:
        self._store = store

    def index_memory_entry(self, *, user_id: str, entry: UserProfileMemoryEntry) -> None:
        if entry.deleted:
            return
        content = (entry.content or "").strip()
        if not content:
            return

        entity_type = entry.kind.value if isinstance(entry.kind, MemoryKind) else str(entry.kind)
        node_id = f"{user_id}:{entry.entry_id}"
        self._store.upsert_node(
            EntityNode(
                entity_id=node_id,
                label=content[:120],
                entity_type=entity_type,
                attributes={"user_id": user_id, "entry_id": entry.entry_id},
            )
        )
        user_node_id = f"user:{user_id}"
        self._store.upsert_node(
            EntityNode(
                entity_id=user_node_id,
                label=user_id,
                entity_type="user",
            )
        )
        self._store.add_edge(
            EntityEdge(
                source_id=user_node_id,
                target_id=node_id,
                relation="has_memory",
                valid_from=entry.valid_from,
                valid_until=entry.valid_until,
            )
        )

    def neighbors_for_user(self, user_id: str) -> list[EntityNode]:
        return self._store.neighbors(f"user:{user_id}")

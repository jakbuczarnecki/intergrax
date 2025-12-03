# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from intergrax.memory.long_term_memory import LongTermMemoryItem, MemoryOwnerRef
from intergrax.memory.long_term_memory_store import LongTermMemoryStore


class InMemoryLongTermMemoryStore(LongTermMemoryStore):
    """
    In-memory implementation of LongTermMemoryStore.

    Use cases:
      - unit tests,
      - local development,
      - small-scale experiments and notebooks.

    This implementation:
      - is NOT durable (process memory only),
      - is NOT suitable for production,
      - uses naive keyword-based search.
    """

    def __init__(self) -> None:
        # item_id -> LongTermMemoryItem
        self._items: Dict[str, LongTermMemoryItem] = {}

    async def upsert_item(self, item: LongTermMemoryItem) -> None:
        """
        Insert or update a long-term memory item.
        """
        self._items[item.item_id] = item

    async def get_item(self, item_id: str) -> Optional[LongTermMemoryItem]:
        """
        Fetch a memory item by its global identifier.
        """
        item = self._items.get(item_id)
        if item is not None:
            item.touch_access()
        return item

    async def delete_item(self, item_id: str) -> None:
        """
        Permanently remove a memory item, if present.
        """
        self._items.pop(item_id, None)

    async def list_items_for_owner(
        self,
        owner: MemoryOwnerRef,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> List[LongTermMemoryItem]:
        """
        List memory items for the given owner.

        Ordering: most recently created first, then by item_id as a tiebreaker.
        """
        owned_items = [
            item for item in self._items.values()
            if item.owner == owner
        ]

        owned_items.sort(
            key=lambda i: (i.created_utc, i.item_id),
            reverse=True,
        )

        sliced = owned_items[offset : offset + limit]
        for item in sliced:
            item.touch_access()
        return sliced

    async def search(
        self,
        owner: MemoryOwnerRef,
        query: str,
        *,
        limit: int = 10,
        min_importance: float = 0.0,
        required_tags: Optional[Sequence[str]] = None,
    ) -> List[LongTermMemoryItem]:
        """
        Very simple reference search implementation:

        - filters by owner,
        - filters by min_importance and required_tags,
        - scores items by the number of query tokens found in `text`,
        - returns top `limit` items.

        Production implementations should replace this with semantic/vector search.
        """
        # Fallback: if query is empty, just list items for the owner.
        if not query:
            return await self.list_items_for_owner(owner, limit=limit)

        tokens = [t for t in query.lower().split() if t]
        if not tokens:
            return await self.list_items_for_owner(owner, limit=limit)

        required_tags = list(required_tags or [])

        candidates: List[LongTermMemoryItem] = []

        for item in self._items.values():
            if item.owner != owner:
                continue
            if item.importance < min_importance:
                continue
            if required_tags and not all(tag in item.tags for tag in required_tags):
                continue

            text_lc = item.text.lower()
            match_count = sum(1 for t in tokens if t in text_lc)
            if match_count == 0:
                continue

            candidates.append(item)

        # Score: simple combination of matches and importance.
        def _score(i: LongTermMemoryItem) -> tuple:
            text_lc = i.text.lower()
            match_count = sum(1 for t in tokens if t in text_lc)
            return (match_count, i.importance, i.created_utc)

        candidates.sort(key=_score, reverse=True)

        result = candidates[:limit]
        for item in result:
            item.touch_access()
        return result

    # Optional helper for debugging / tests
    def list_all_item_ids(self) -> List[str]:
        """
        Return all item_ids currently stored in memory.
        """
        return list(self._items.keys())

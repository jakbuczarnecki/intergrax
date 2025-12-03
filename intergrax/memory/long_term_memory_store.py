# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence

from intergrax.memory.long_term_memory import LongTermMemoryItem, MemoryOwnerRef


class LongTermMemoryStore(Protocol):
    """
    Persistent storage interface for long-term memory items.

    This abstraction sits *under* higher-level mechanisms such as:
      - ingestion pipelines,
      - vectorstores / semantic search layers,
      - summarization and promotion to user/organization profiles.

    Responsibilities:
      - store and retrieve `LongTermMemoryItem` aggregates,
      - provide a basic search API scoped by owner.

    It MUST NOT:
      - call LLMs directly,
      - implement prompt construction,
      - implement RAG orchestration.
    """

    async def upsert_item(self, item: LongTermMemoryItem) -> None:
        """
        Insert or update a long-term memory item.

        If an item with the same `item_id` already exists, it MUST be replaced.
        """
        ...

    async def get_item(self, item_id: str) -> Optional[LongTermMemoryItem]:
        """
        Fetch a memory item by its global identifier.

        Returns:
            - LongTermMemoryItem if found,
            - None if no such item exists.
        """
        ...

    async def delete_item(self, item_id: str) -> None:
        """
        Permanently remove a memory item.

        Implementations MUST tolerate unknown item_ids without raising errors.
        """
        ...

    async def list_items_for_owner(
        self,
        owner: MemoryOwnerRef,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> List[LongTermMemoryItem]:
        """
        List memory items for the given owner.

        Ordering is implementation-specific; it MAY be:
          - by recency,
          - by importance,
          - by created_utc,
        as long as it is deterministic within a single store implementation.
        """
        ...

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
        Search long-term memory for items relevant to a textual query,
        scoped to the given owner.

        Implementations MAY:
          - use naive keyword search (reference implementation),
          - use vector similarity search,
          - use hybrid approaches.

        The API is intentionally general enough to support semantic search backends.
        """
        ...

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Iterable, List, Optional, Sequence

from intergrax.memory.long_term_memory import (
    LongTermMemoryItem,
    MemoryOwnerRef,
)
from intergrax.memory.long_term_memory_store import LongTermMemoryStore


class LongTermMemoryManager:
    """
    High-level façade for working with long-term memory.

    Responsibilities:
      - provide convenient, semantically meaningful methods for:
          * saving session summaries,
          * saving explicit "remember this" notes,
          * saving distilled facts from documents,
          * searching for context before answering a user query;
      - hide the low-level details of `LongTermMemoryItem` construction.

    It intentionally does NOT:
      - call LLMs directly,
      - decide *when* to write memory (that is a policy concern handled by the runtime),
      - implement RAG or prompt building.
    """

    def __init__(self, store: LongTermMemoryStore) -> None:
        self._store = store

    # -------------------------------------------------------------------------
    # Item construction helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _generate_item_id() -> str:
        """
        Generate a globally unique identifier for a memory item.

        Using UUID4 keeps this independent from storage backends
        and avoids collisions across processes.
        """
        return str(uuid.uuid4())

    # -------------------------------------------------------------------------
    # Public write APIs
    # -------------------------------------------------------------------------

    async def save_manual_note(
        self,
        *,
        owner: MemoryOwnerRef,
        text: str,
        tags: Optional[Iterable[str]] = None,
        importance: float = 0.7,
        source_reference: str = "",
        metadata: Optional[dict[str, str]] = None,
    ) -> LongTermMemoryItem:
        """
        Save an explicit, user/admin "remember this" note.

        Typical trigger:
          - user says: "remember this",
          - UI has a "pin to memory" action.

        The caller is expected to provide already-compact text.
        """
        item = LongTermMemoryItem(
            item_id=self._generate_item_id(),
            owner=owner,
            text=text,
            source_type="manual_note",
            source_reference=source_reference,
            importance=importance,
            tags=list(tags or []),
            metadata=dict(metadata or {}),
        )
        await self._store.upsert_item(item)
        return item

    async def save_conversation_summary(
        self,
        *,
        owner: MemoryOwnerRef,
        summary_text: str,
        session_id: str,
        tags: Optional[Iterable[str]] = None,
        importance: float = 0.6,
        metadata: Optional[dict[str, str]] = None,
    ) -> LongTermMemoryItem:
        """
        Save a summary of a conversation/session as a long-term memory item.

        Typical usage:
          - at the end of a session, the runtime asks an LLM to produce
            a compact summary of the key decisions, preferences and insights,
          - this method persists that summary as a single LTM entry.
        """
        base_tags = ["source:conversation"]
        if tags:
            base_tags.extend(tags)

        meta = dict(metadata or {})
        meta.setdefault("session_id", session_id)
        meta.setdefault("summary_created_utc", datetime.utcnow().isoformat())

        item = LongTermMemoryItem(
            item_id=self._generate_item_id(),
            owner=owner,
            text=summary_text,
            source_type="conversation",
            source_reference=session_id,
            importance=importance,
            tags=base_tags,
            metadata=meta,
        )
        await self._store.upsert_item(item)
        return item

    async def save_document_fact(
        self,
        *,
        owner: MemoryOwnerRef,
        fact_text: str,
        document_id: str,
        tags: Optional[Iterable[str]] = None,
        importance: float = 0.5,
        metadata: Optional[dict[str, str]] = None,
    ) -> LongTermMemoryItem:
        """
        Save a distilled fact or insight that originates from a document.

        This is typically used when an ingestion pipeline or a document
        QA/analysis flow promotes important knowledge from documents
        into long-term memory.
        """
        base_tags = ["source:document"]
        if tags:
            base_tags.extend(tags)

        meta = dict(metadata or {})
        meta.setdefault("document_id", document_id)

        item = LongTermMemoryItem(
            item_id=self._generate_item_id(),
            owner=owner,
            text=fact_text,
            source_type="document",
            source_reference=document_id,
            importance=importance,
            tags=base_tags,
            metadata=meta,
        )
        await self._store.upsert_item(item)
        return item

    # -------------------------------------------------------------------------
    # Read / search APIs
    # -------------------------------------------------------------------------

    async def search_for_context(
        self,
        *,
        owner: MemoryOwnerRef,
        query: str,
        limit: int = 5,
        min_importance: float = 0.0,
        required_tags: Optional[Sequence[str]] = None,
    ) -> List[LongTermMemoryItem]:
        """
        Search long-term memory for items relevant to the given query,
        scoped to the provided owner.

        This is the main entrypoint for ContextBuilder / runtime when
        building context before an LLM call.
        """
        return await self._store.search(
            owner=owner,
            query=query,
            limit=limit,
            min_importance=min_importance,
            required_tags=required_tags,
        )

    async def get_recent_items(
        self,
        *,
        owner: MemoryOwnerRef,
        limit: int = 20,
        offset: int = 0,
    ) -> List[LongTermMemoryItem]:
        """
        Retrieve recent long-term memory items for the given owner.

        This can be useful for:
          - debugging,
          - building a "memory inspector" UI,
          - providing context for summarization or compaction jobs.
        """
        return await self._store.list_items_for_owner(
            owner=owner,
            limit=limit,
            offset=offset,
        )

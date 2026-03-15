# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Context builder for nexus Mode.

This module is responsible for:
- Deciding whether RAG should be used for a given request.
- Retrieving relevant document chunks from the vector store for the current session.
- Providing:
    * a RAG-specific system prompt,
    * a list of retrieved chunks,
    * debug metadata for observability.

Design principles:
- ContextBuilder does NOT own or build conversation history.
  Conversation history is managed by SessionStore and composed by the runtime engine.
- ContextBuilder is ignorant of:
    * LLM adapter details,
    * how messages are serialized for OpenAI/Gemini/Claude,
    * how RouteInfo is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from typing import TYPE_CHECKING

from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from intergrax.llm.messages import ChatMessage
if TYPE_CHECKING:
    from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.chat_session import ChatSession


@dataclass
class RetrievedChunk:
    """
    Lightweight representation of a single retrieved document chunk.

    This is an internal structure used by nexus Mode.
    It wraps whatever the underlying vector store returns into a
    stable shape that can be:
    - injected into prompts,
    - exposed in debug traces,
    - later used for citations.
    """

    id: str
    text: str
    metadata: Dict[str, Any]
    score: float


@dataclass
class BuiltContext:
    """
    Result of ContextBuilder.build_context(...).

    This object is consumed by the runtime engine and prompt builders.
    - history_messages: conversation history built by the runtime / SessionStore.
      ContextBuilder does not build or trim history; it only passes through
      the list it receives from the engine.
    - retrieved_chunks: RAG context (can be serialized into prompt).
    - rag_used: whether RAG context should be considered "used" (i.e., non-empty chunks).
    - rag_reason: decision/reason string (stable, non-dict contract).
    """
    history_messages: List[ChatMessage]
    retrieved_chunks: List[RetrievedChunk]
    rag_used: bool
    rag_reason: str


class ContextBuilder:
    """
    Build RAG-related context for nexus Mode.

    Responsibilities:
    - Decide whether to use RAG for a given (session, request).
    - Retrieve relevant document chunks from the vector store using
      session/user/tenant/workspace metadata.

    This class does NOT:
    - build or trim conversation history,
    - know anything about tools,
    - know anything about user/organization profiles.
    """

    def __init__(
        self,
        config: "RuntimeConfig",
        vectorstore_manager: VectorstoreManager,
        *,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Args:
            config: nexus Mode runtime configuration.
            vectorstore_manager: Shared vector store manager instance.
            collection_name: Optional explicit collection/index name.
                If None, the manager's default collection should be used.
        """
        self._config = config
        self._vectorstore = vectorstore_manager
        self._collection_name = collection_name

    
    async def build_context(
        self,
        session: ChatSession,
        request: RuntimeRequest,
        base_history: List[ChatMessage]
    ) -> BuiltContext:
        """
        High-level orchestration method.

        Steps:
        1. Receive base conversation history (already built/reduced by the runtime).
        2. Decide whether RAG should be used for this request.
        3. If yes, retrieve document chunks from the vector store.
        4. Compose a RAG-specific system prompt (for now: DEFAULT_SYSTEM_PROMPT).
        5. Return BuiltContext with:
            - system_prompt,
            - reduced history_messages,
            - retrieved_chunks,
            - structured RAG debug info.

        Important:
        - Conversation history comes from the ChatSession, which is populated
          by SessionStore. ContextBuilder does NOT own any persistence layer.
        """

        # 1. Decide whether we should use RAG for this request
        use_rag, rag_reason = self._should_use_rag(session, request)

        retrieved_chunks: List[RetrievedChunk] = []
        retrieval_reason: Optional[str] = None

        if use_rag:
            retrieved_chunks, retrieval_reason = self._retrieve_for_session(session, request)
            if retrieval_reason:
                rag_reason = retrieval_reason
            elif not retrieved_chunks:
                rag_reason = "no_hits"
        else:
            retrieved_chunks = []

        rag_used = bool(retrieved_chunks)

        return BuiltContext(
            history_messages=list(base_history or []),
            retrieved_chunks=retrieved_chunks,
            rag_used=rag_used,
            rag_reason=rag_reason,
        )


    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _should_use_rag(
        self,
        session: ChatSession,
        request: RuntimeRequest,
    ) -> Tuple[bool, str]:
        """
        Decide whether to use RAG for this request.

        Current policy (intentionally simple and predictable):

        - If RAG is disabled in the runtime config -> do not use it.
        - If RAG is enabled -> always query the vector store.

        Whether any chunks are actually retrieved depends on vector store
        contents and metadata filters.

        More sophisticated heuristics (e.g. based on attachments or message type)
        can be added later without changing the engine API.
        """
        if not self._config.enable_rag:
            return False, "rag_disabled_in_config"

        return True, "rag_enabled_in_config"

    
    def _retrieve_for_session(
        self,
        session: ChatSession,
        request: RuntimeRequest,
    ) -> Tuple[List[RetrievedChunk], Optional[str]]:
        """
        Perform a vector store query for this session.
        Returns: (retrieved_chunks, retrieval_reason).
        retrieval_reason is None when retrieval was attempted successfully.
        """

        # 1) Build the logical `where` based on session and request metadata        
        query_text = request.message
        query_text = str(query_text or "")

        # Base metadata filters – this keeps all chunks that belong to this
        # logical conversation scope (session/user/tenant/workspace).
        where: Dict[str, Any] = {}
        for attr in ("id", "user_id", "tenant_id", "workspace_id"):            
            value = getattr(session, attr) if hasattr(session, attr) else None
            if value is not None:
                # We normalize "id" to "session_id" for clarity in the metadata.
                if attr == "id":
                    where["session_id"] = value
                else:
                    where[attr] = value

        # NOTE:
        # We intentionally do NOT filter by a single attachment_id here, because
        # RuntimeRequest currently exposes attachments as a list[AttachmentRef],
        # not as a single "attachment_id". At this stage we want to retrieve
        # all chunks for the given session/user/tenant/workspace.
        #
        # In the future, when the attachment model is fully stabilized, you can
        # extend this method to support additional scoping such as:
        # - "only chunks for the last uploaded attachment",
        # - "only chunks for a specific AttachmentRef.id",
        # based on request.attachments or request.metadata.

        max_docs: int = int(self._config.max_docs_per_query)
        score_threshold: Optional[float] = self._config.rag_score_threshold

        metadata_filter = MetadataFilter(conditions=where) if where else None

        # 2) Get embedding manager from runtime config
        embedding_manager = self._config.embedding_manager
        if embedding_manager is None:
            # Without an embedding manager we cannot perform semantic search.
            return [], "no_embedding_manager_in_config"

        # 3) Compute query embedding using IntergraxEmbeddingManager API
        try:
            query_embedding = embedding_manager.embed_one(query_text)
        except Exception:
            query_embedding = embedding_manager.embed_texts([query_text])

        # Normalize embeddings shape for vector store:
        # - numpy array: convert 2D batch-of-1 into 1D vector
        # - plain list: unwrap batch-of-1 into a single vector
        if hasattr(query_embedding, "ndim"):
            try:
                if query_embedding.ndim > 1:
                    query_embedding = query_embedding[0]
            except Exception:
                pass
        elif (
            isinstance(query_embedding, (list, tuple))
            and query_embedding
            and isinstance(query_embedding[0], (list, tuple))
        ):
            query_embedding = query_embedding[0]

        # 4) Call vector store with the normalized Tier-1 contract
        hits = self._vectorstore.query(
            query_embedding=query_embedding,
            top_k=max_docs,
            metadata_filter=metadata_filter,
            include_embeddings=False,
        )

        # 5) Normalize hits into RetrievedChunk objects
        retrieved_chunks = self._map_hits_to_chunks(hits)

        # Apply score_threshold as an extra safety net
        if score_threshold is not None:
            filtered_chunks: List[RetrievedChunk] = []
            for ch in retrieved_chunks:
                if ch.score >= score_threshold:
                    filtered_chunks.append(ch)
            retrieved_chunks = filtered_chunks

        # 6) Build RAG debug info (backend-agnostic view)            
        return retrieved_chunks, None


    def _build_backend_where(self, where: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Translate a simple metadata dict into a backend-compatible `where` filter.

        For a Chroma-like backend we produce:

            {
                "$and": [
                    {"session_id": {"$eq": "..."}},
                    {"user_id": {"$eq": "..."}},
                    ...
                ]
            }

        If the input dict is empty, returns None (no filter).
        """
        if not where:
            return None

        conditions: List[Dict[str, Any]] = []
        for key, value in where.items():
            if value is None:
                continue
            conditions.append({key: {"$eq": value}})

        if not conditions:
            return None

        return {"$and": conditions}

    def _map_hits_to_chunks(self, hits: List[VectorStoreHit]) -> List[RetrievedChunk]:
        """
        Convert Tier-1 VectorStoreHit objects into RetrievedChunk objects.
        """

        if not hits:
            return []

        chunks: List[RetrievedChunk] = []

        for hit in hits:

            metadata = dict(hit.metadata or {})

            raw_id = hit.id or "unknown"

            raw_text = hit.content or ""

            try:
                score = float(hit.similarity_score)
            except Exception:
                score = 0.0

            chunks.append(
                RetrievedChunk(
                    id=str(raw_id),
                    text=str(raw_text),
                    metadata=metadata,
                    score=score,
                )
            )

        return chunks

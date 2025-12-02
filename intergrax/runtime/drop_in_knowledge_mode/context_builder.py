# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Context builder for Drop-In Knowledge Mode.

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

from intergrax.rag.vectorstore_manager import VectorstoreManager
from intergrax.llm.messages import ChatMessage

from .config import RuntimeConfig
from .response_schema import RuntimeRequest
from .session_store import ChatSession


# RAG-specific default system prompt.
# Global / user-profile instructions will be added by the runtime engine
# (not by this module).
DEFAULT_SYSTEM_PROMPT = (
    "You are an AI assistant running in the Intergrax Drop-In Knowledge Mode.\n\n"
    "The runtime automatically:\n"
    "- receives and processes any user attachments (files),\n"
    "- splits them into chunks, indexes them in a vector store,\n"
    "- retrieves only the most relevant chunks for the current question.\n\n"
    "You see the final retrieved context as plain text snippets. You do NOT need "
    "to talk about indexing, vector stores, embeddings, or technical ingestion steps.\n\n"
    "Important behavior rules:\n"
    "1) Never say that you cannot see or access attachments. The runtime has already "
    "processed them for you.\n"
    "2) Never mention internal markers or identifiers from the retrieved context "
    "(such as IDs, scores, or file paths). They are for internal use only.\n"
    "3) Use the retrieved context when (and only when) it is relevant to the user's question.\n"
    "4) If the answer is not present in the retrieved context nor in the conversation history, "
    "say so explicitly instead of inventing details.\n"
    "5) Focus on the user's intent rather than on the mechanics of the system.\n"
    "6) If the user explicitly asks you to 'index', 'upload', 'attach', or 'remember' a file, "
    "assume that this has already been done by the runtime and simply confirm in one or two "
    "sentences that the file is indexed and now available as context. Do NOT explain ingestion "
    "modules, vector stores, or internal runtime components in your answer.\n"
)


@dataclass
class RetrievedChunk:
    """
    Lightweight representation of a single retrieved document chunk.

    This is an internal structure used by Drop-In Knowledge Mode.
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

    This object is consumed by the runtime engine and prompt builders:
    - system_prompt: RAG-related system message for the LLM.
    - history_messages: conversation history built by the runtime / SessionStore.
      ContextBuilder does not build or trim history; it only passes through
      the list it receives from the engine.
    - retrieved_chunks: RAG context (can be serialized into prompt).
    - rag_debug_info: structured debug trace to be surfaced in
      RuntimeAnswer.debug_trace["rag"].
    """

    system_prompt: str
    history_messages: List[ChatMessage]
    retrieved_chunks: List[RetrievedChunk]
    rag_debug_info: Dict[str, Any]


class ContextBuilder:
    """
    Build RAG-related context for Drop-In Knowledge Mode.

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
        config: RuntimeConfig,
        vectorstore_manager: VectorstoreManager,
        *,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Args:
            config: Drop-In Knowledge Mode runtime configuration.
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

        if use_rag:
            retrieved_chunks, rag_debug_info = self._retrieve_for_session(session, request)
        else:
            # No RAG for this request – keep debug info explicit so it is easy
            # to see in RuntimeAnswer.debug_trace why RAG was skipped.
            retrieved_chunks = []
            rag_debug_info = {
                "enabled": bool(getattr(self._config, "enable_rag", True)),
                "used": False,
                "reason": rag_reason,
                "hits_count": 0,
                "where_filter": {},
                "top_k": int(getattr(self._config, "max_docs_per_query", 5)),
                "score_threshold": getattr(self._config, "rag_score_threshold", None),
                "hits": [],
            }

        # 2. RAG-related system prompt.
        #    Even jeśli w danym requestcie retrieved_chunks jest puste,
        #    runtime nadal faktycznie indeksuje załączniki i może korzystać
        #    z RAG, więc ten system prompt może zostać globalnie.
        system_prompt = getattr(self._config, "system_prompt", None) or DEFAULT_SYSTEM_PROMPT

        return BuiltContext(
            system_prompt=system_prompt,
            history_messages=list(base_history or []),
            retrieved_chunks=retrieved_chunks,
            rag_debug_info=rag_debug_info,
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
    ) -> Tuple[List[RetrievedChunk], Dict[str, Any]]:
        """
        Perform a vector store query for this session.

        Strategy:
        - Build a logical `where` filter using:
            * session_id
            * user_id
            * tenant_id
            * workspace_id
          (optionally you can extend this in the future with additional
           filters derived from request.metadata or attachments).
        - Use `request.message` as the query text.
        - Compute query embeddings via the configured embedding manager.
        - Call `IntergraxVectorstoreManager.query(...)` with `query_embeddings`.
        """
        # 1) Build the logical `where` based on session and request metadata
        query_text = getattr(request, "message", None) or getattr(request, "query", "")
        query_text = str(query_text or "")

        # Base metadata filters – this keeps all chunks that belong to this
        # logical conversation scope (session/user/tenant/workspace).
        where: Dict[str, Any] = {}
        for attr in ("id", "user_id", "tenant_id", "workspace_id"):
            value = getattr(session, attr, None)
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

        max_docs: int = int(getattr(self._config, "max_docs_per_query", 5))
        score_threshold: Optional[float] = getattr(self._config, "rag_score_threshold", None)

        # Translate logical `where` into backend-specific filter
        backend_where = self._build_backend_where(where)

        # 2) Get embedding manager from runtime config
        embedding_manager = getattr(self._config, "embedding_manager", None)
        if embedding_manager is None:
            # Without an embedding manager we cannot perform semantic search.
            # We return an empty result with a clear diagnostic reason.
            return [], {
                "enabled": self._config.enable_rag,
                "used": False,
                "reason": "no_embedding_manager_in_config",
                "where_filter": where,
                "top_k": max_docs,
                "score_threshold": score_threshold,
                "hits": [],
            }

        # 3) Compute query embeddings using IntergraxEmbeddingManager API
        # Preferred path: single-text embedding
        try:
            query_embeddings = embedding_manager.embed_one(query_text)
        except Exception:
            # Fallback: some providers might only support batch embedding
            query_embeddings = embedding_manager.embed_texts([query_text])

        # Normalize embeddings shape for vector store:
        # - numpy array: ensure 2D
        # - plain list: wrap 1D into batch-of-1
        if hasattr(query_embeddings, "ndim"):
            try:
                if query_embeddings.ndim == 1:
                    query_embeddings = query_embeddings.reshape(1, -1)
            except Exception:
                pass
        else:
            if isinstance(query_embeddings, (list, tuple)) and query_embeddings:
                first = query_embeddings[0]
                if isinstance(first, (float, int)):
                    # 1D -> wrap into batch
                    query_embeddings = [query_embeddings]

        # 4) Call vector store with embeddings + backend_where
        hits_dict = self._vectorstore.query(
            query_embeddings=query_embeddings,
            top_k=max_docs,
            where=backend_where,
            include_embeddings=False,
        )

        # 5) Normalize hits into RetrievedChunk objects
        retrieved_chunks = self._map_hits_to_chunks(hits_dict)

        # Apply score_threshold as an extra safety net
        if score_threshold is not None:
            filtered_chunks: List[RetrievedChunk] = []
            for ch in retrieved_chunks:
                if ch.score >= score_threshold:
                    filtered_chunks.append(ch)
            retrieved_chunks = filtered_chunks

        # 6) Build RAG debug info (backend-agnostic view)
        rag_used = bool(retrieved_chunks)

        rag_debug_info: Dict[str, Any] = {
            "enabled": getattr(self._config, "enable_rag", False),
            "used": rag_used,
            "hits_count": len(retrieved_chunks or []),
            "where_filter": where,
            "top_k": max_docs,
            "score_threshold": score_threshold,
        }

        # Only store full hit metadata if something was actually retrieved
        if rag_used:
            rag_debug_info["hits"] = [
                {
                    "id": ch.id,
                    "score": round(ch.score, 4),
                    "metadata": ch.metadata,
                    "preview": ch.text[:200],
                }
                for ch in retrieved_chunks
            ]
        else:
            rag_debug_info["hits"] = []

        return retrieved_chunks, rag_debug_info


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

    def _map_hits_to_chunks(self, hits: Any) -> List[RetrievedChunk]:
        """
        Normalize raw hits from the vector store into RetrievedChunk objects.

        Supported patterns:
        - Dict with parallel lists (Chroma-style).
        - Dict with a `matches` list.
        - Flat list or list-of-lists of dict-like objects.
        """
        if not hits:
            return []

        if isinstance(hits, dict):
            if "matches" in hits and isinstance(hits["matches"], list):
                flat_hits = hits["matches"]
            else:
                docs = (
                    hits.get("documents")
                    or hits.get("texts")
                    or hits.get("contents")
                )
                metas = hits.get("metadatas") or hits.get("metadata")
                scores = hits.get("scores") or hits.get("distances")
                ids = hits.get("ids") or hits.get("id")

                if docs is None:
                    return []

                if isinstance(docs, list) and docs and isinstance(docs[0], list):
                    docs_list = docs[0]
                    metas_list = metas[0] if isinstance(metas, list) and metas else metas
                    scores_list = scores[0] if isinstance(scores, list) and scores else scores
                    ids_list = ids[0] if isinstance(ids, list) and ids else ids
                else:
                    docs_list = docs
                    metas_list = metas
                    scores_list = scores
                    ids_list = ids

                flat_hits = []
                n = len(docs_list)

                for i in range(n):
                    doc_text = docs_list[i]

                    if isinstance(metas_list, (list, tuple)) and i < len(metas_list):
                        meta_i = metas_list[i]
                    else:
                        meta_i = metas_list or {}

                    if isinstance(scores_list, (list, tuple)) and i < len(scores_list):
                        score_i = scores_list[i]
                    else:
                        score_i = scores_list

                    if isinstance(ids_list, (list, tuple)) and i < len(ids_list):
                        id_i = ids_list[i]
                    else:
                        id_i = ids_list

                    flat_hits.append(
                        {
                            "id": id_i,
                            "text": doc_text,
                            "metadata": meta_i,
                            "distance": score_i,
                        }
                    )
        else:
            if isinstance(hits, list) and hits and isinstance(hits[0], list):
                flat_hits = hits[0]
            else:
                flat_hits = hits

        chunks: List[RetrievedChunk] = []

        for raw in flat_hits:
            if not isinstance(raw, dict):
                raw_dict = getattr(raw, "__dict__", {}) or {}
            else:
                raw_dict = raw

            metadata = raw_dict.get("metadata") or raw_dict.get("meta") or {}
            if not isinstance(metadata, dict):
                metadata = {"_raw_metadata": metadata}

            raw_id = (
                raw_dict.get("id")
                or raw_dict.get("doc_id")
                or metadata.get("id")
                or metadata.get("doc_id")
                or "unknown"
            )

            raw_text = (
                raw_dict.get("text")
                or raw_dict.get("page_content")
                or raw_dict.get("content")
                or ""
            )

            raw_score = raw_dict.get("score")
            if raw_score is None:
                raw_score = raw_dict.get("distance")
                if raw_score is not None:
                    try:
                        raw_score = 1.0 / (1.0 + float(raw_score))
                    except Exception:
                        raw_score = 0.0

            try:
                score = float(raw_score) if raw_score is not None else 0.0
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

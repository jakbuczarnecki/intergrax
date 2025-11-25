# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Context builder for Drop-In Knowledge Mode.

This module is responsible for:
- Deciding whether RAG should be used for a given request.
- Retrieving relevant document chunks from the vector store for the current session.
- Building:
    * system prompt (for now: simple default, later: dedicated builder),
    * reduced chat history,
    * list of retrieved chunks + debug info.

It is intentionally minimal. The goal of this first iteration:
- Provide a clean extension point for RAG.
- Be easy to integrate into `DropInKnowledgeRuntime.ask()` in the next step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from intergrax.rag.vectorstore_manager import IntergraxVectorstoreManager
from intergrax.llm.conversational_memory import ChatMessage

from .config import RuntimeConfig
from .response_schema import RuntimeRequest
from .session_store import ChatSession


# For now we keep a very simple default prompt here.
# In a later step we will move to a dedicated system_prompts module
# and build a richer, configurable system prompt.
DEFAULT_SYSTEM_PROMPT =  (
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

    This object is meant to be consumed by the runtime engine:
    - system_prompt: system message for the LLM.
    - history_messages: reduced chat history (already in ChatMessage format).
    - retrieved_chunks: RAG context (can be serialized into prompt).
    - rag_debug_info: structured debug trace to be surfaced in RuntimeAnswer.debug_trace["rag"].
    """

    system_prompt: str
    history_messages: List[ChatMessage]
    retrieved_chunks: List[RetrievedChunk]
    rag_debug_info: Dict[str, Any]


class ContextBuilder:
    """
    Build LLM-ready context for Drop-In Knowledge Mode.

    Responsibilities:
    - Decide whether to use RAG for a given (session, request).
    - Retrieve relevant document chunks from the vector store
      using session/user/tenant metadata.
    - Reduce chat history to a manageable window.
    - Return a BuiltContext object that the engine can translate into
      OpenAI-style messages.

    This class is intentionally ignorant of:
    - The exact LLM adapter implementation.
    - How the final prompt is serialized.
    - How RouteInfo is built.

    Those concerns stay in the runtime engine.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        vectorstore_manager: IntergraxVectorstoreManager,
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
    ) -> BuiltContext:
        """
        High-level orchestration method.

        Steps:
        1. Build reduced chat history (last N messages).
        2. Decide whether RAG should be used.
        3. If yes, retrieve document chunks from the vector store.
        4. Compose system prompt (for now: DEFAULT_SYSTEM_PROMPT).
        5. Return BuiltContext with debug info for RAG.

        This method is async to allow future integration with
        async vector stores or other async dependencies.
        Currently the implementation is synchronous inside.
        """
        history_messages = self._build_history(session)

        use_rag, rag_reason = self._should_use_rag(session, request)

        if use_rag:
            retrieved_chunks, rag_debug_info = self._retrieve_for_session(session, request)
        else:
            retrieved_chunks = []
            rag_debug_info = {
                "used": False,
                "reason": rag_reason,
                "hits": [],
            }

        system_prompt = getattr(self._config, "system_prompt", None) or DEFAULT_SYSTEM_PROMPT

        return BuiltContext(
            system_prompt=system_prompt,
            history_messages=history_messages,
            retrieved_chunks=retrieved_chunks,
            rag_debug_info=rag_debug_info,
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _build_history(self, session: ChatSession) -> List[ChatMessage]:
        """
        Build reduced chat history for the LLM.

        Strategy (minimal, token-agnostic):
        - Take session.messages (or [] if missing).
        - Keep only the last N messages, where N comes from
          config.max_history_messages (fallback: 20).

        In a later iteration we can:
        - Switch to token-aware truncation.
        - Filter out tool messages if needed.
        """
        messages: Sequence[ChatMessage] = getattr(session, "messages", []) or []
        max_history: int = int(getattr(self._config, "max_history_messages", 20))

        if max_history <= 0:
            return []

        if len(messages) <= max_history:
            return list(messages)

        return list(messages[-max_history:])

    # Heuristic version
    # def _should_use_rag(
    #     self,
    #     session: ChatSession,
    #     request: RuntimeRequest,
    # ) -> Tuple[bool, str]:
    #     """
    #     Decide whether to use RAG for this request.

    #     Minimal strategy:
    #     - If config.disable_rag / not enable_rag → False.
    #     - If there are no attachments in this session → False.
    #     - Otherwise → True.

    #     The exact way we detect attachments depends on the
    #     ChatSession / ChatMessage schema. For now we try a couple
    #     of common patterns and keep the logic conservative.
    #     """
    #     enable_rag: bool = bool(getattr(self._config, "enable_rag", True))
    #     if not enable_rag:
    #         return False, "rag_disabled_in_config"

    #     # Heuristic: look for attachments on session and messages.
    #     # Adapt this logic to your actual schema if needed.
    #     session_attachments = getattr(session, "attachments", None)
    #     if session_attachments:
    #         return True, "attachments_present_on_session"

    #     messages: Sequence[ChatMessage] = getattr(session, "messages", []) or []
    #     for msg in messages:
    #         if getattr(msg, "attachments", None):
    #             return True, "attachments_present_on_messages"

    #     # Optionally: if request explicitly references an attachment_id,
    #     # we can treat that as a hint to use RAG.
    #     attachment_id = getattr(request, "attachment_id", None)
    #     if attachment_id is not None:
    #         return True, "attachment_id_present_on_request"

    #     return False, "no_attachments_detected"
    def _should_use_rag(
        self,
        session: ChatSession,
        request: RuntimeRequest,
    ) -> Tuple[bool, str]:
        """
        Decide whether to use RAG for this request.

        For now we keep the policy intentionally simple:

        - If RAG is disabled in the runtime config -> do not use it.
        - If RAG is enabled -> always try to use it.

        This guarantees that:
        - the vector store is always queried when `enable_rag=True`,
        - whether any chunks are actually retrieved depends on metadata filters
        and the contents of the vector store.

        More sophisticated heuristics (e.g. checking attachments on the session
        or the request) can be added later when the message/attachment model
        is stable across different session stores.
        """
        enable_rag: bool = bool(getattr(self._config, "enable_rag", True))
        if not enable_rag:
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
            * optional attachment_id from the request
        - Use `request.message` as the query text.
        - Compute query embeddings via the configured embedding manager.
        - Call `IntergraxVectorstoreManager.query(...)` with `query_embeddings`.

        NOTE:
        -----
        This implementation matches the actual signature:

            query(
                query_embeddings: NDArray[np.float32] | Sequence[Sequence[float]],
                top_k: int = 5,
                *,
                where: Optional[Dict[str, Any]] = None,
                include_embeddings: bool = False,
            ) -> Dict[str, Any]
        """
        # 1) Build the logical `where` based on session and request metadata
        query_text = getattr(request, "message", None) or getattr(request, "query", "")
        query_text = str(query_text)

        where: Dict[str, Any] = {}

        # Robust extraction of session metadata
        session_id = getattr(session, "id", None) or getattr(session, "session_id", None)
        if session_id is not None:
            where["session_id"] = session_id

        for attr in ("user_id", "tenant_id", "workspace_id"):
            value = getattr(session, attr, None)
            if value is not None:
                where[attr] = value

        attachment_id = getattr(request, "attachment_id", None)
        if attachment_id is not None:
            where["attachment_id"] = attachment_id

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
                "used": False,
                "reason": "no_embedding_manager_in_config",
                "where_filter": where,
                "top_k": max_docs,
                "score_threshold": score_threshold,
            }

        # 3) Compute query embeddings using IntergraxEmbeddingManager API
        # Preferred path: single-text embedding
        try:
            query_embeddings = embedding_manager.embed_one(query_text)
        except Exception:
            # Fallback: some providers might only support batch calls
            query_embeddings = embedding_manager.embed_texts([query_text])

        # Ensure a 2D structure for the vector store (batch with size 1)
        # Handle both numpy arrays and plain Python sequences.
        if hasattr(query_embeddings, "ndim"):
            # Likely a numpy array
            if query_embeddings.ndim == 1:
                # Shape (dim,) -> (1, dim)
                query_embeddings = query_embeddings.reshape(1, -1)
        else:
            # Fallback: Python list/tuple
            if isinstance(query_embeddings, (list, tuple)) and query_embeddings and isinstance(
                query_embeddings[0], (float, int)
            ):
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
        rag_debug_info: Dict[str, Any] = {
            "used": True,
            # Logical where – the one we reason about and test against
            "where_filter": where,
            "top_k": max_docs,
            "score_threshold": score_threshold,
            "hits": [
                {
                    "id": ch.id,
                    "score": ch.score,
                    "metadata": ch.metadata,
                    "preview": ch.text[:200],
                }
                for ch in retrieved_chunks
            ],
        }

        return retrieved_chunks, rag_debug_info





    def _build_backend_where(self, where: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Translate a simple metadata dict into a backend-compatible `where` filter.

        Logical input used by ContextBuilder:

            {
                "session_id": "...",
                "user_id": "...",
                "tenant_id": "...",
                "workspace_id": "...",
                ...
            }

        For the current Chroma backend, we produce:

            {
                "$and": [
                    {"session_id": {"$eq": "..."}},
                    {"user_id": {"$eq": "..."}},
                    ...
                ]
            }

        If the input dict is empty, returns None (no filter).

        NOTE:
        -----
        The name intentionally does not mention Chroma. If you ever switch
        to a different vector store backend, this is the only place that
        needs to be adjusted.
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

        1. Dict with parallel lists (typical IntergraxVectorstoreManager / Chroma output), e.g.:

        {
            "ids": [[...]],
            "documents": [[...]],
            "metadatas": [[...]],
            "distances": [[...]],
        }

        2. Dict with a `matches` list (some vector DB clients):

        {
            "matches": [
                {"id": ..., "score": ..., "metadata": {...}, "text": ...},
                ...
            ]
        }

        3. A flat list of dicts or a list-of-lists (fallback for other adapters).

        Everything is converted into a flat list of `RetrievedChunk`.
        """
        if not hits:
            return []

        # Case 1: dict-based API
        if isinstance(hits, dict):
            # 1a) "matches" key with list of dicts
            if "matches" in hits and isinstance(hits["matches"], list):
                flat_hits = hits["matches"]
            else:
                # 1b) Parallel lists: ids, documents/texts, metadatas, scores/distances
                docs = (
                    hits.get("documents")
                    or hits.get("texts")
                    or hits.get("contents")
                )
                metas = hits.get("metadatas") or hits.get("metadata")
                scores = hits.get("scores") or hits.get("distances")
                ids = hits.get("ids") or hits.get("id")

                if docs is None:
                    # We do not know how to interpret this structure
                    return []

                # Chroma-style 2D structure: [[doc1, doc2, ...]]
                if isinstance(docs, list) and docs and isinstance(docs[0], list):
                    docs_list = docs[0]
                    metas_list = metas[0] if isinstance(metas, list) and metas else metas
                    scores_list = scores[0] if isinstance(scores, list) and scores else scores
                    ids_list = ids[0] if isinstance(ids, list) and ids else ids
                else:
                    # Already flat lists
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
                            # traktujemy to jako "distance" – score przeliczymy niżej
                            "distance": score_i,
                        }
                    )
        else:
            # Case 2: list / list-of-lists – keep old behavior, flatten if needed
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
                    # Many vector DBs use distance (smaller is better).
                    # Convert to a simple relevance-like score.
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



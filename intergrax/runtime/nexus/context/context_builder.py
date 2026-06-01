# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Nexus session RAG context builder (Tier-1).

Retrieval for the LLM pipeline is performed in ``RagStep`` / ``rag.retrieve`` when
``perform_retrieval=False`` (default for ``HistoryStep``). See Phase Q-R.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from intergrax.llm.messages import ChatMessage
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.resolve import resolve_retrieval_service
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

if TYPE_CHECKING:
    from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.chat_session import ChatSession


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float


@dataclass
class BuiltContext:
    history_messages: List[ChatMessage]
    retrieved_chunks: List[RetrievedChunk]
    rag_used: bool
    rag_reason: str


class ContextBuilder:
    """Build RAG-related context for nexus Mode (history pass-through + optional retrieval)."""

    def __init__(
        self,
        config: "RuntimeConfig",
        vectorstore_manager: VectorstoreManager,
        *,
        collection_name: Optional[str] = None,
    ) -> None:
        self._config = config
        self._vectorstore = vectorstore_manager
        self._collection_name = collection_name

    async def build_context(
        self,
        session: ChatSession,
        request: RuntimeRequest,
        base_history: List[ChatMessage],
        *,
        perform_retrieval: bool = True,
    ) -> BuiltContext:
        use_rag, rag_reason = self._should_use_rag(session, request)

        retrieved_chunks: List[RetrievedChunk] = []
        retrieval_reason: Optional[str] = None

        if use_rag and perform_retrieval:
            retrieved_chunks, retrieval_reason = self._retrieve_for_session(session, request)
            if retrieval_reason:
                rag_reason = retrieval_reason
            elif not retrieved_chunks:
                rag_reason = "no_hits"
        elif not use_rag:
            rag_reason = rag_reason if rag_reason else "rag_not_requested"

        rag_used = bool(retrieved_chunks)

        return BuiltContext(
            history_messages=list(base_history or []),
            retrieved_chunks=retrieved_chunks,
            rag_used=rag_used,
            rag_reason=rag_reason,
        )

    def _should_use_rag(
        self,
        session: ChatSession,
        request: RuntimeRequest,
    ) -> Tuple[bool, str]:
        if not self._config.enable_rag:
            return False, "rag_disabled_in_config"

        meta = request.metadata or {}
        if meta.get("use_rag") is False:
            return False, "rag_disabled_in_request_metadata"
        if meta.get("use_rag") is True:
            return True, "rag_enabled_in_request_metadata"

        return True, "rag_enabled_in_config"

    def _retrieve_for_session(
        self,
        session: ChatSession,
        request: RuntimeRequest,
    ) -> Tuple[List[RetrievedChunk], Optional[str]]:
        query_text = str(request.message or "")

        where: Dict[str, Any] = {}
        for attr in ("id", "user_id", "tenant_id", "workspace_id"):
            value = getattr(session, attr) if hasattr(session, attr) else None
            if value is not None:
                if attr == "id":
                    where["session_id"] = value
                else:
                    where[attr] = value

        max_docs: int = int(self._config.max_docs_per_query)
        score_threshold: Optional[float] = self._config.rag_score_threshold

        metadata_filter = MetadataFilter(conditions=where) if where else None

        embedding_manager = self._config.embedding_manager
        if embedding_manager is None:
            return [], "no_embedding_manager_in_config"

        profile = self._config.rag_profile or RagProfile(final_top_k=max_docs)
        service = self._config.retrieval_service
        if service is None:
            service = resolve_retrieval_service(
                vectorstore_manager=self._vectorstore,
                embedding_manager=embedding_manager,
                retriever_manager=self._config.retriever_manager,
                reranker_manager=self._config.reranker_manager,
                profile=profile,
            )
        if service is None:
            return [], "retrieval_service_not_configured"

        result = service.retrieve(
            RetrievalRequest(
                query=query_text,
                final_top_k=max_docs,
                metadata_filter=metadata_filter,
                score_threshold=score_threshold,
            )
        )
        if not result.used:
            return [], result.reason

        retrieved_chunks = [
            RetrievedChunk(
                id=c.id,
                text=c.text,
                metadata=dict(c.metadata or {}),
                score=c.score,
            )
            for c in result.chunks
        ]
        return retrieved_chunks, None

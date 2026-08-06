# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Nexus session RAG context builder (Tier-1).

Retrieval for the LLM context is performed via ``rag.retrieve`` (catalog) in ``on_next_step`` when
``perform_retrieval=False`` (default for ``HistoryStep``). See Phase Q-R.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from intergrax.llm.messages import ChatMessage
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreScope,
)

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
        vectorstore_manager: BaseVectorstoreManager,
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

        from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID

        allowed_tools = meta.get("allowed_tools")
        if isinstance(allowed_tools, (list, tuple, set)):
            if RAG_RETRIEVE_TOOL_ID in allowed_tools:
                return True, "rag_via_allowed_tools"
            if allowed_tools and RAG_RETRIEVE_TOOL_ID not in allowed_tools:
                return False, "rag_not_in_allowed_tools"

        tool_ids = meta.get("tool_ids")
        if isinstance(tool_ids, (list, tuple, set)):
            if RAG_RETRIEVE_TOOL_ID in tool_ids:
                return True, "rag_via_tool_ids"
            if tool_ids and RAG_RETRIEVE_TOOL_ID not in tool_ids:
                return False, "rag_not_in_tool_ids"

        return True, "rag_enabled_in_config"

    def _retrieve_for_session(
        self,
        session: ChatSession,
        request: RuntimeRequest,
    ) -> Tuple[List[RetrievedChunk], Optional[str]]:
        query_text = str(request.message or "")

        scope, scope_reason = self._resolve_scope(session, request)
        if scope_reason:
            return [], scope_reason
        assert scope is not None

        where: Dict[str, Any] = {}
        if session.id:
            where["session_id"] = session.id
        if session.user_id is not None:
            where["user_id"] = session.user_id

        max_docs: int = int(self._config.max_docs_per_query)
        score_threshold: Optional[float] = self._config.rag_score_threshold

        metadata_filter = MetadataFilter(conditions=where) if where else None

        embedding_manager = self._config.embedding_manager
        if embedding_manager is None:
            return [], "no_embedding_manager_in_config"

        profile = self._config.rag_profile or RagProfile(final_top_k=max_docs)
        service = self._config.retrieval_service
        if service is None:
            from intergrax.rag.retrieval.resolve import (
                resolve_retrieval_service,
            )

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
                scope=scope,
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

    def _resolve_scope(
        self,
        session: ChatSession,
        request: RuntimeRequest,
    ) -> Tuple[Optional[VectorStoreScope], Optional[str]]:
        def canonical(value: object) -> str | None:
            if value is None:
                return None
            normalized = str(value).strip()
            return normalized or None

        def agree(values: tuple[str | None, ...]) -> str | None:
            supplied = {value for value in values if value is not None}
            return next(iter(supplied)) if len(supplied) == 1 else None

        tenant_values = (
            canonical(request.tenant_id),
            canonical(self._config.tenant_id),
            canonical(session.tenant_id),
        )
        tenants = {value for value in tenant_values if value is not None}
        if len(tenants) > 1:
            return None, "tenant_scope_conflict"
        tenant_id = agree(tenant_values)
        if tenant_id is None:
            return None, "tenant_scope_required"

        workspace_values = (
            canonical(request.workspace_id),
            canonical(self._config.workspace_id),
            canonical(session.workspace_id),
        )
        workspaces = {value for value in workspace_values if value is not None}
        if len(workspaces) > 1:
            return None, "workspace_scope_conflict"
        workspace_id = agree(workspace_values)

        bound_scope = getattr(self._vectorstore, "bound_scope", None)
        if not isinstance(bound_scope, VectorStoreScope):
            bound_scope = None
        if bound_scope is not None:
            if tenant_id != bound_scope.tenant_id:
                return None, "tenant_scope_conflict"
            if bound_scope.workspace_id is not None:
                if workspace_id is not None and workspace_id != bound_scope.workspace_id:
                    return None, "workspace_scope_conflict"
                workspace_id = bound_scope.workspace_id

        return (
            VectorStoreScope(
                tenant_id=tenant_id,
                namespace=bound_scope.namespace if bound_scope is not None else None,
                workspace_id=workspace_id,
            ),
            None,
        )


SessionRagContextBuilder = ContextBuilder
"""Deprecated alias — use ``ContextBuilder`` (CE-3.6)."""

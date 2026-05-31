# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Retrieval logic for ``rag.retrieve`` — uses Tier-0 :class:`RetrievalService`."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.resolve import resolve_retrieval_service
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_result import RetrievalChunk
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter
from intergrax.tools.providers.rag.contracts import RagChunkResult, RagRetrieveInput, RagRetrieveOutput
from intergrax.tools.registry.wiring import ToolWiringContext

RAG_TOOL_ID = "rag.retrieve"


class RagRetrieveConfigurationError(RuntimeError):
    """Raised when wiring context lacks required RAG dependencies."""


def perform_rag_retrieve(ctx: ToolWiringContext, params: RagRetrieveInput) -> RagRetrieveOutput:
    """
    Retrieve document chunks via unified :class:`RetrievalService` (hybrid + rerank per profile).
    """
    if ctx.vectorstore_manager is None:
        return RagRetrieveOutput(used=False, reason="vectorstore_manager_not_configured")
    if ctx.embedding_manager is None:
        return RagRetrieveOutput(used=False, reason="embedding_manager_not_configured")

    profile = ctx.rag_profile or RagProfile()
    service = resolve_retrieval_service(
        vectorstore_manager=ctx.vectorstore_manager,
        embedding_manager=ctx.embedding_manager,
        retriever_manager=ctx.retriever_manager,
        reranker_manager=ctx.reranker_manager,
        profile=profile,
        retrieval_service=ctx.retrieval_service,
    )
    if service is None:
        return RagRetrieveOutput(used=False, reason="retrieval_service_not_configured")

    where = _build_metadata_scope(params)
    metadata_filter = MetadataFilter(conditions=where) if where else None

    result = service.retrieve(
        RetrievalRequest(
            query=params.query,
            top_k=int(params.top_k) if params.top_k else None,
            metadata_filter=metadata_filter,
            score_threshold=params.score_threshold,
        )
    )

    if not result.used:
        return RagRetrieveOutput(used=False, reason=result.reason)

    chunks = [_to_rag_chunk(c) for c in result.chunks]
    max_chars = profile.max_context_chars
    context_text = format_rag_context_text(chunks, max_chars=max_chars)
    diagnostics = {
        "retriever_id": result.trace.retriever_id,
        "route_tier": result.trace.route_tier,
        "reranker_id": result.trace.reranker_id,
        "rerank_enabled": result.trace.rerank_enabled,
        "retrieval_latency_ms": result.trace.retrieval_latency_ms,
        "rerank_latency_ms": result.trace.rerank_latency_ms,
    }
    return RagRetrieveOutput(
        used=True,
        chunks=chunks,
        context_text=context_text,
        reason="ok",
        diagnostics=diagnostics,
    )


def _to_rag_chunk(c: RetrievalChunk) -> RagChunkResult:
    return RagChunkResult(
        id=c.id,
        text=c.text,
        score=c.score,
        metadata=dict(c.metadata or {}),
    )


def _build_metadata_scope(params: RagRetrieveInput) -> dict[str, Any]:
    where: dict[str, Any] = {}
    if params.session_id is not None:
        where["session_id"] = params.session_id
    if params.user_id is not None:
        where["user_id"] = params.user_id
    if params.tenant_id is not None:
        where["tenant_id"] = params.tenant_id
    if params.workspace_id is not None:
        where["workspace_id"] = params.workspace_id
    return where


def format_rag_context_text(
    chunks: Sequence[RagChunkResult],
    *,
    max_chars: int = 4000,
) -> str:
    """Compact, LLM-friendly preview of retrieved chunks."""
    lines: List[str] = []
    total = 0

    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata or {}
        header_parts: List[str] = [f"[{idx}]"]
        src = meta.get("source") or meta.get("url") or meta.get("doc_id") or meta.get("file")
        if src:
            header_parts.append(str(src))
        page = meta.get("page") or meta.get("page_number")
        if page is not None:
            header_parts.append(f"p={page}")

        block = " ".join(header_parts) + "\n" + chunk.text.strip()
        if not block.strip():
            continue

        if total + len(block) + 2 > max_chars:
            remaining = max_chars - total
            if remaining > 80:
                lines.append(block[:remaining].rstrip() + "…")
            break

        lines.append(block)
        total += len(block) + 2

    return "\n\n".join(lines).strip()

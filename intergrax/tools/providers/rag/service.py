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
from intergrax.rag.retrieval.citation import Citation
from intergrax.tools.providers.rag.contracts import (
    RagChunkResult,
    RagCitationResult,
    RagRetrieveInput,
    RagRetrieveOutput,
)
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
    citations = [_to_rag_citation(c) for c in result.citations]
    chunks, citations, poisoning_reason, poisoning_warnings = _apply_retrieval_poisoning_filter(
        ctx,
        chunks,
        citations,
    )
    if not chunks:
        return RagRetrieveOutput(
            used=False,
            reason=poisoning_reason or "retrieval_poisoning_quarantine",
            diagnostics={"poisoning_review_warnings": poisoning_warnings},
        )

    max_chars = profile.max_context_chars
    context_text = format_rag_context_text(chunks, max_chars=max_chars)
    diagnostics = {
        "retriever_id": result.trace.retriever_id,
        "route_tier": result.trace.route_tier,
        "reranker_id": result.trace.reranker_id,
        "rerank_enabled": result.trace.rerank_enabled,
        "retrieval_latency_ms": result.trace.retrieval_latency_ms,
        "rerank_latency_ms": result.trace.rerank_latency_ms,
        "citation_count": len(citations),
        "embedding_version_filtered_count": result.trace.embedding_version_filtered_count,
    }
    if result.trace.embedding_version_warnings:
        diagnostics["embedding_version_warnings"] = list(result.trace.embedding_version_warnings)
    if poisoning_warnings:
        diagnostics["poisoning_review_warnings"] = poisoning_warnings
    if poisoning_reason:
        diagnostics["poisoning_quarantine_applied"] = True
    return RagRetrieveOutput(
        used=True,
        chunks=chunks,
        citations=citations,
        context_text=context_text,
        reason=poisoning_reason or "ok",
        diagnostics=diagnostics,
    )


def _retrieval_poisoning_defense_enabled(ctx: ToolWiringContext) -> bool:
    profile = ctx.security_profile
    if profile is None:
        raw = ctx.extras.get("security_profile")
        profile = raw if raw is not None else None
    if profile is None:
        return False
    return bool(getattr(profile, "retrieval_poisoning_defense_enabled", False))


def _apply_retrieval_poisoning_filter(
    ctx: ToolWiringContext,
    chunks: List[RagChunkResult],
    citations: List[RagCitationResult],
) -> tuple[List[RagChunkResult], List[RagCitationResult], str, List[str]]:
    if not chunks or not _retrieval_poisoning_defense_enabled(ctx):
        return chunks, citations, "", []

    from intergrax.runtime.architecture.retrieval_security_wiring import (
        filter_retrieved_chunks_for_poisoning,
    )
    from intergrax.runtime.nexus.context.context_builder import RetrievedChunk

    retrieved = [
        RetrievedChunk(
            id=chunk.id,
            text=chunk.text,
            metadata=dict(chunk.metadata or {}),
            score=chunk.score,
        )
        for chunk in chunks
    ]
    filtered, warnings = filter_retrieved_chunks_for_poisoning(retrieved)
    if len(filtered) == len(retrieved):
        return chunks, citations, "", warnings

    allowed_ids = {chunk.id for chunk in filtered}
    filtered_chunks = [chunk for chunk in chunks if chunk.id in allowed_ids]
    filtered_citations = [citation for citation in citations if citation.chunk_id in allowed_ids]
    reason = "retrieval_poisoning_quarantine" if not filtered_chunks else "ok"
    return filtered_chunks, filtered_citations, reason, warnings


def _to_rag_chunk(c: RetrievalChunk) -> RagChunkResult:
    return RagChunkResult(
        id=c.id,
        text=c.text,
        score=c.score,
        metadata=dict(c.metadata or {}),
    )


def _to_rag_citation(citation: Citation) -> RagCitationResult:
    return RagCitationResult(
        chunk_id=citation.chunk_id,
        source_id=citation.source_id,
        source_type=citation.source_type,
        source_label=citation.source_label,
        url=citation.url,
        page=citation.page,
        score=citation.score,
        excerpt=citation.excerpt,
        metadata=dict(citation.metadata or {}),
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

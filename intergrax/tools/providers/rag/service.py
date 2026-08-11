# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Retrieval logic for ``rag.retrieve`` — uses Tier-0 :class:`RetrievalService`."""

from __future__ import annotations
from intergrax.utils import attribute_access

from typing import Any, List, Sequence

from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.knowledge.contracts.validation import knowledge_metadata_to_plain
from intergrax.rag.retrieval.resolve import resolve_retrieval_service
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_result import RetrievalChunk
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataMembershipCondition,
    VectorStoreScope,
)
from intergrax.rag.retrieval.citation import Citation
from intergrax.tools.providers.rag.contracts import (
    RagChunkResult,
    RagCitationResult,
    RagRetrieveInput,
    RagRetrieveOutput,
)
from intergrax.tools.providers.rag.scope import (
    authoritative_tenant_id,
    resolve_tenant_scoped_vectorstore,
    use_wired_retrieval_managers,
)
from intergrax.tools.registry.wiring import ToolWiringContext

RAG_TOOL_ID = "rag.retrieve"


def _validate_routing_identifier(
    value: object | None,
    *,
    invalid_reason: str,
) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    if not isinstance(value, str):
        return None, invalid_reason
    normalized = value.strip()
    if not normalized:
        return None, invalid_reason
    return normalized, None


class RagRetrieveConfigurationError(RuntimeError):
    """Raised when wiring context lacks required RAG dependencies."""


def perform_rag_retrieve(ctx: ToolWiringContext, params: RagRetrieveInput) -> RagRetrieveOutput:
    """
    Retrieve document chunks via unified :class:`RetrievalService` (hybrid + rerank per profile).
    """
    request_tenant, tenant_error = _validate_routing_identifier(
        params.tenant_id,
        invalid_reason="tenant_scope_invalid",
    )
    if tenant_error:
        return RagRetrieveOutput(used=False, reason=tenant_error)
    tenant_id, tenant_conflict = authoritative_tenant_id(request_tenant=request_tenant)
    if tenant_conflict:
        return RagRetrieveOutput(used=False, reason=tenant_conflict)

    workspace_id, workspace_error = _validate_routing_identifier(
        params.workspace_id,
        invalid_reason="workspace_scope_invalid",
    )
    if workspace_error:
        return RagRetrieveOutput(used=False, reason=workspace_error)

    configured_scope = attribute_access.optional(ctx.vectorstore_manager, "bound_scope", None)
    if configured_scope is not None and not isinstance(configured_scope, VectorStoreScope):
        return RagRetrieveOutput(used=False, reason="tenant_scope_invalid")
    if configured_scope is not None and tenant_id is None:
        tenant_id = configured_scope.tenant_id
        if configured_scope.workspace_id is not None:
            if (
                workspace_id is not None
                and workspace_id != configured_scope.workspace_id
            ):
                return RagRetrieveOutput(used=False, reason="workspace_scope_conflict")
            workspace_id = configured_scope.workspace_id

    if tenant_id is None:
        return RagRetrieveOutput(used=False, reason="tenant_scope_required")

    vectorstore = resolve_tenant_scoped_vectorstore(ctx, tenant_id)
    if vectorstore is None:
        return RagRetrieveOutput(used=False, reason="vectorstore_manager_not_configured")
    if ctx.embedding_manager is None:
        return RagRetrieveOutput(used=False, reason="embedding_manager_not_configured")

    profile = ctx.rag_profile or RagProfile()
    wired_retrieval = use_wired_retrieval_managers(ctx, vectorstore)
    service = resolve_retrieval_service(
        vectorstore_manager=vectorstore,
        embedding_manager=ctx.embedding_manager,
        retriever_manager=ctx.retriever_manager if wired_retrieval else None,
        reranker_manager=ctx.reranker_manager,
        profile=profile,
        retrieval_service=ctx.retrieval_service if wired_retrieval else None,
    )
    if service is None:
        return RagRetrieveOutput(used=False, reason="retrieval_service_not_configured")

    bound_scope = attribute_access.optional(vectorstore, "bound_scope", None)
    if bound_scope is not None and not isinstance(bound_scope, VectorStoreScope):
        return RagRetrieveOutput(used=False, reason="tenant_scope_invalid")
    if isinstance(bound_scope, VectorStoreScope):
        if tenant_id is not None and bound_scope.tenant_id != tenant_id:
            return RagRetrieveOutput(used=False, reason="tenant_scope_conflict")
        if bound_scope.workspace_id is not None:
            if workspace_id is not None and workspace_id != bound_scope.workspace_id:
                return RagRetrieveOutput(used=False, reason="workspace_scope_conflict")
            workspace_id = bound_scope.workspace_id
        tenant_id = bound_scope.tenant_id
    operation_scope = (
        VectorStoreScope(
            tenant_id=tenant_id,
            namespace=bound_scope.namespace if bound_scope is not None else None,
            workspace_id=workspace_id,
        )
        if tenant_id is not None
        else None
    )
    metadata_filter = _build_metadata_filter(params)

    result = service.retrieve(
        RetrievalRequest(
            query=params.query,
            top_k=int(params.top_k) if params.top_k else None,
            scope=operation_scope,
            metadata_filter=metadata_filter,
            score_threshold=params.score_threshold,
        )
    )

    if not result.used:
        diagnostics: dict[str, Any] = {}
        if result.trace.retrieval_error_kind:
            diagnostics["retrieval_error_kind"] = result.trace.retrieval_error_kind
        if result.trace.attempted_retriever_ids:
            diagnostics["attempted_retriever_ids"] = list(result.trace.attempted_retriever_ids)
        return RagRetrieveOutput(
            used=False,
            reason=result.reason,
            diagnostics=diagnostics,
        )

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
        "route_classifier": result.trace.route_classifier,
        "citation_count": len(citations),
        "embedding_version_filtered_count": result.trace.embedding_version_filtered_count,
    }
    if result.trace.embedding_version_warnings:
        diagnostics["embedding_version_warnings"] = list(result.trace.embedding_version_warnings)
    if result.trace.channel_contributions:
        diagnostics["channel_contributions"] = {
            channel: list(chunk_ids)
            for channel, chunk_ids in result.trace.channel_contributions.items()
        }
    if result.trace.graph_provenance_records:
        diagnostics["graph_provenance_records"] = list(result.trace.graph_provenance_records)
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
    return bool(attribute_access.optional(profile, "retrieval_poisoning_defense_enabled", False))


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
    metadata = knowledge_metadata_to_plain(c.metadata or {})
    metadata.setdefault("document_id", c.id)
    return RagChunkResult(
        id=c.id,
        text=c.text,
        score=c.score,
        rank=c.rank,
        channel=c.channel,
        vector_id=c.vector_id,
        scope=dict(c.scope),
        provenance=dict(c.provenance),
        user_metadata=knowledge_metadata_to_plain(c.user_metadata or metadata),
        metadata=metadata,
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
        metadata=knowledge_metadata_to_plain(citation.metadata or {}),
    )


def _build_metadata_scope(params: RagRetrieveInput) -> dict[str, Any]:
    where: dict[str, Any] = {}
    if params.session_id is not None:
        where["session_id"] = params.session_id
    if params.user_id is not None:
        where["user_id"] = params.user_id
    return where


def _build_metadata_filter(params: RagRetrieveInput) -> MetadataFilter | None:
    where = _build_metadata_scope(params)
    membership: tuple[MetadataMembershipCondition, ...] = ()
    if params.allowed_source_ids:
        membership = (
            MetadataMembershipCondition(
                field="source_id",
                allowed_values=tuple(params.allowed_source_ids),
            ),
        )
    if not where and not membership:
        return None
    return MetadataFilter(conditions=where, membership=membership)


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

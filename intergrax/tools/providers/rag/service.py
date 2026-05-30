# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Retrieval logic for ``rag.retrieve`` — composes Tier-0 RAG managers from wiring context."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit
from intergrax.tools.providers.rag.contracts import RagChunkResult, RagRetrieveInput, RagRetrieveOutput
from intergrax.tools.registry.wiring import ToolWiringContext

RAG_TOOL_ID = "rag.retrieve"


class RagRetrieveConfigurationError(RuntimeError):
    """Raised when wiring context lacks required RAG dependencies."""


def perform_rag_retrieve(ctx: ToolWiringContext, params: RagRetrieveInput) -> RagRetrieveOutput:
    """
    Retrieve document chunks for ``params.query`` using vector store + embeddings.

    Requires ``ctx.vectorstore_manager`` and ``ctx.embedding_manager``.
    """
    vectorstore = ctx.vectorstore_manager
    embedding_manager = ctx.embedding_manager

    if vectorstore is None:
        return RagRetrieveOutput(used=False, reason="vectorstore_manager_not_configured")
    if embedding_manager is None:
        return RagRetrieveOutput(used=False, reason="embedding_manager_not_configured")

    where = _build_metadata_scope(params)
    metadata_filter = MetadataFilter(conditions=where) if where else None

    query_embedding = _embed_query(embedding_manager, params.query)
    hits = vectorstore.query(
        query_embedding=query_embedding,
        top_k=int(params.top_k),
        metadata_filter=metadata_filter,
        include_embeddings=False,
    )

    chunks = _map_hits(hits)
    if params.score_threshold is not None:
        threshold = float(params.score_threshold)
        chunks = [chunk for chunk in chunks if chunk.score >= threshold]

    if not chunks:
        return RagRetrieveOutput(used=False, reason="no_hits")

    context_text = format_rag_context_text(chunks)
    return RagRetrieveOutput(
        used=True,
        chunks=chunks,
        context_text=context_text,
        reason="ok",
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


def _embed_query(embedding_manager: Any, query_text: str) -> Sequence[float]:
    try:
        query_embedding = embedding_manager.embed_one(query_text)
    except Exception:
        query_embedding = embedding_manager.embed_texts([query_text])

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
    return query_embedding


def _map_hits(hits: List[VectorStoreHit]) -> List[RagChunkResult]:
    chunks: List[RagChunkResult] = []
    for hit in hits or []:
        if not isinstance(hit, VectorStoreHit):
            continue
        text = (hit.content or "").strip()
        if not text:
            continue
        try:
            score = float(hit.similarity_score)
        except Exception:
            score = 0.0
        chunks.append(
            RagChunkResult(
                id=str(hit.id or "unknown"),
                text=text,
                score=score,
                metadata=dict(hit.metadata or {}),
            )
        )
    return chunks


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

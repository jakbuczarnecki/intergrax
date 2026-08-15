# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.rerankers.bootstrap.reranker_bootstrap import create_default_reranker_manager
from intergrax.rag.rerankers.contracts.reranker_types import RerankerCandidate
from intergrax.tools.providers.rag.rerank_contracts import RagRerankChunkOutput, RagRerankInput, RagRerankOutput
from intergrax.tools.registry.wiring import ToolWiringContext

RAG_RERANK_TOOL_ID = "rag.rerank"


def rag_rerank(ctx: ToolWiringContext, params: RagRerankInput) -> RagRerankOutput:
    profile = ctx.rag_profile or RagProfile()
    reranker_manager = ctx.reranker_manager
    if reranker_manager is None and profile.enable_rerank:
        reranker_manager = create_default_reranker_manager()
    if reranker_manager is None:
        raise RuntimeError("reranker_manager_not_configured")

    candidates = tuple(
        RerankerCandidate(
            document=KnowledgeDocument(
                schema_version=1,
                identity={
                    "document_id": chunk.id or f"rag-tool-{index}",
                    "root_document_id": chunk.id or f"rag-tool-{index}",
                },
                scope={"tenant_id": "rag_tool"},
                content=chunk.text,
                metadata={
                    key: value
                    for key, value in chunk.metadata.items()
                    if value is not None
                    and key
                    not in {
                        "schema_version",
                        "document_id",
                        "root_document_id",
                        "parent_document_id",
                        "tenant_id",
                        "namespace",
                        "source_kind",
                        "source_id",
                        "source_parent_id",
                        "provider_id",
                        "source_revision",
                        "source_uri",
                        "content_hash",
                    }
                },
                provenance={
                    "source_kind": "rag_tool",
                    "source_id": chunk.id or f"rag-tool-{index}",
                },
            ),
            original_score=chunk.score if chunk.score is not None else 0.0,
            original_rank=index,
            channel="rag_tool",
        )
        for index, chunk in enumerate(params.chunks)
    )
    results = reranker_manager.rerank(query=params.query.strip(), candidates=candidates, limit=params.top_n)
    reranker_id = profile.reranker_id or "default"
    output_chunks = [
        RagRerankChunkOutput(
            id=item.candidate.document.identity.document_id,
            text=item.candidate.document.content,
            score=item.rerank_score,
            rank=item.rank,
            metadata=dict(item.candidate.document.metadata),
        )
        for item in results
    ]
    return RagRerankOutput(
        query=params.query.strip(),
        chunks=output_chunks,
        reranker_id=reranker_id,
        total=len(output_chunks),
    )

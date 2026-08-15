# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from intergrax.rag.ingest.ingest_pipeline import IngestPipeline, IngestRequest
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.tools.providers.rag.ingest_contracts import RagIngestInput, RagIngestOutput
from intergrax.tools.providers.rag.source_operation_wiring import (
    bind_source_operation_coordinator,
    shared_source_operation_coordinator,
)
from intergrax.tools.providers.rag.scope import (
    TENANT_ID_METADATA_CONFLICT,
    authoritative_tenant_id,
    resolve_tenant_scoped_vectorstore,
)
from intergrax.tools.registry.wiring import ToolWiringContext

RAG_INGEST_TOOL_ID = "rag.ingest_document"


def perform_rag_ingest(ctx: ToolWiringContext, params: RagIngestInput) -> RagIngestOutput:
    tenant_id, tenant_conflict = authoritative_tenant_id(
        request_tenant=params.tenant_id,
        metadata_tenant=params.metadata.get("tenant_id"),
    )
    if tenant_conflict:
        return RagIngestOutput(used=False, reason=tenant_conflict)

    vectorstore = resolve_tenant_scoped_vectorstore(ctx, tenant_id)
    embedding_manager = ctx.embedding_manager
    if vectorstore is None or embedding_manager is None:
        return RagIngestOutput(used=False, reason="vectorstore_or_embedding_not_configured")

    bind_source_operation_coordinator(ctx, vectorstore)

    path = Path(params.source_path)
    if not path.exists():
        return RagIngestOutput(used=False, reason="source_not_found")

    loader = ctx.extras.get("documents_loader")
    splitter = ctx.extras.get("documents_splitter")
    if loader is None:
        from intergrax.rag.document_loaders.bootstrap.default_loader import create_default_documents_loader

        loader = create_default_documents_loader()
    if splitter is None:
        from intergrax.rag.document_splitters.bootstrap.default_chunking_engine import (
            create_default_document_splitter,
        )

        splitter = create_default_document_splitter()

    profile = ctx.rag_profile or RagProfile()
    pipeline = IngestPipeline(
        loader=loader,
        splitter=splitter,
        embedding_manager=embedding_manager,
        vectorstore=vectorstore,
        toc_vectorstore=ctx.toc_vectorstore_manager,
        profile=profile,
        contextual_enricher=ctx.extras.get("contextual_enricher"),
        graph_store=ctx.extras.get("graph_store"),
        llm_for_graph=ctx.extras.get("llm_adapter"),
        source_coordinator=shared_source_operation_coordinator(ctx),
    )

    base_metadata: dict[str, Any] = dict(params.metadata)
    if params.session_id is not None:
        base_metadata["session_id"] = params.session_id
    if params.user_id is not None:
        base_metadata["user_id"] = params.user_id
    if tenant_id is not None:
        base_metadata["tenant_id"] = tenant_id

    strategy_id: Optional[str] = base_metadata.pop("chunking_strategy_id", None)

    result = pipeline.run(
        IngestRequest(
            source_path=str(path),
            base_metadata=base_metadata,
            chunking_strategy_id=strategy_id,
            workspace_id=params.workspace_id,
        )
    )

    if not result.used:
        return RagIngestOutput(
            used=False,
            reason=result.reason,
            parser_id=result.parser_id,
            parser_trace=result.parser_trace,
            file_size_bytes=result.file_size_bytes,
            async_job_recommended=result.async_job_recommended,
        )

    return RagIngestOutput(
        used=True,
        num_chunks=result.num_chunks,
        vector_ids=result.vector_ids,
        reason=result.reason,
        parser_id=result.parser_id,
        parser_trace=result.parser_trace,
        file_size_bytes=result.file_size_bytes,
        async_job_recommended=result.async_job_recommended,
    )

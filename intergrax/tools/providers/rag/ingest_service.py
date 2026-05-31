# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pathlib import Path
from typing import Any

from intergrax.rag.document_loaders.pipeline.parser_pipeline import TRACE_METADATA_KEY
from intergrax.tools.providers.rag.ingest_contracts import RagIngestInput, RagIngestOutput
from intergrax.tools.registry.wiring import ToolWiringContext

RAG_INGEST_TOOL_ID = "rag.ingest_document"


def perform_rag_ingest(ctx: ToolWiringContext, params: RagIngestInput) -> RagIngestOutput:
    vectorstore = ctx.vectorstore_manager
    embedding_manager = ctx.embedding_manager
    if vectorstore is None or embedding_manager is None:
        return RagIngestOutput(used=False, reason="vectorstore_or_embedding_not_configured")

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

    base_metadata: dict[str, Any] = dict(params.metadata)
    if params.session_id is not None:
        base_metadata["session_id"] = params.session_id
    if params.user_id is not None:
        base_metadata["user_id"] = params.user_id
    if params.tenant_id is not None:
        base_metadata["tenant_id"] = params.tenant_id
    if params.workspace_id is not None:
        base_metadata["workspace_id"] = params.workspace_id

    def _metadata_callback(_doc, _source: str) -> dict[str, Any]:
        return dict(base_metadata)

    docs = loader.load_document(
        str(path),
        use_default_metadata=True,
        call_custom_metadata=_metadata_callback,
    )
    if not docs:
        return RagIngestOutput(used=False, reason="no_documents_loaded")

    parser_trace: dict[str, Any] = {}
    parser_id: str | None = None
    first_meta = docs[0].metadata or {}
    if TRACE_METADATA_KEY in first_meta:
        parser_trace = dict(first_meta.get(TRACE_METADATA_KEY) or {})
        parser_id = parser_trace.get("parser_id") or first_meta.get("integration_parser_id")

    chunks = splitter.split_documents(docs)
    if not chunks:
        return RagIngestOutput(
            used=False,
            reason="no_chunks_generated",
            parser_id=parser_id,
            parser_trace=parser_trace,
        )

    try:
        embed_result = embedding_manager.embed_documents(chunks)
        aligned_docs = embed_result.documents
        embeddings = embed_result.embeddings
    except AttributeError:
        texts = [c.page_content for c in chunks]
        embeddings = embedding_manager.embed_texts(texts)
        aligned_docs = chunks

    for doc in aligned_docs:
        doc.metadata = {**(doc.metadata or {}), **base_metadata}

    ids = [f"ingest-{path.stem}-{i}" for i in range(len(aligned_docs))]
    stored_ids = vectorstore.add_documents(
        documents=aligned_docs,
        embeddings=embeddings,
        ids=ids,
        base_metadata=base_metadata,
    )
    vector_ids = list(stored_ids) if stored_ids is not None else ids

    return RagIngestOutput(
        used=True,
        num_chunks=len(aligned_docs),
        vector_ids=vector_ids,
        parser_id=parser_id,
        parser_trace=parser_trace,
        reason="ok",
    )

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Attachment ingestion pipeline for nexus Mode.

This module defines a high-level service that:
  - takes AttachmentRef objects (from sessions/messages),
  - resolves them to loader-compatible paths via AttachmentResolver,
  - loads and splits documents using Intergrax RAG components:
      * IntergraxDocumentsLoader
      * IntergraxDocumentsSplitter
  - embeds them and stores them in a vector database via:
      * IntergraxEmbeddingManager
      * IntergraxVectorstoreManager

The goal is to reuse existing Intergrax RAG building blocks while providing
a clean, runtime-oriented API that operates on AttachmentRef.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document

from intergrax.knowledge.contracts import KnowledgeDocument

from intergrax.llm.messages import AttachmentRef
from intergrax.rag.document_loaders.compat.legacy_runtime_document import (
    to_legacy_rag_document,
)
from intergrax.rag.document_loaders.contracts.base_document_loader import BaseDocumentsLoader
from intergrax.rag.document_splitters.contracts.base_documents_splitter import BaseDocumentsSplitter
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.ingestion_events import record_ingestion_failed
from intergrax.runtime.nexus.context.context_builder import RetrievedChunk
from intergrax.runtime.nexus.ingestion.attachments import AttachmentResolver
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceWriter



# ---------------------------------------------------------------------------
# Ingestion result model
# ---------------------------------------------------------------------------

@dataclass
class IngestionResult:
    """
    Summary information about ingestion of a single attachment.
    """

    attachment_id: str
    attachment_type: str
    num_chunks: int
    vector_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Attachment ingestion service (Intergrax-native)
# ---------------------------------------------------------------------------

class AttachmentIngestionService:
    """
    High-level ingestion service for nexus Mode.

    Responsibilities:
      - Resolve AttachmentRef objects into filesystem Paths (via AttachmentResolver).
      - Load documents using IntergraxDocumentsLoader.load_document(...).
      - Split them into chunks using IntergraxDocumentsSplitter.split_documents(...).
      - Embed chunks (via IntergraxEmbeddingManager).
      - Store vectors (via IntergraxVectorstoreManager).
      - Return a structured IngestionResult per attachment.

    This service does NOT:
      - manage ChatSession objects,
      - perform retrieval or answering.

    It is intended to be called from AgentEngine / host intake or other
    orchestration layers when new attachments are added to a session.
    """

    def __init__(
        self,
        *,
        resolver: AttachmentResolver,
        embedding_manager: BaseEmbeddingManager,
        vectorstore_manager: BaseVectorstoreManager,
        loader: Optional[BaseDocumentsLoader] = None,
        splitter: Optional[BaseDocumentsSplitter] = None,
        trace_writer: Optional[RunTraceWriter] = None,
        event_bus: Optional[RuntimeEventBus] = None,
    ) -> None:
        """
        Args:
            resolver:
                Component that knows how to resolve AttachmentRef.uri into a local Path.
            embedding_manager:
                IntergraxEmbeddingManager used to generate embeddings.
            vectorstore_manager:
                IntergraxVectorstoreManager used to store embeddings + metadata.
            loader:
                Optional custom IntergraxDocumentsLoader instance. If None, a default
                instance is created with conservative settings.
            splitter:
                Optional custom IntergraxDocumentsSplitter instance. If None, a default
                instance is created with standard chunking parameters.
        """
        self._resolver = resolver
        self._embedding_manager = embedding_manager
        self._vectorstore_manager = vectorstore_manager

        # Use provided loader/splitter or fall back to default instances.
        if loader is None:
            from intergrax.rag.document_loaders.bootstrap.default_loader import create_default_documents_loader

            loader = create_default_documents_loader()
        if splitter is None:
            from intergrax.rag.document_splitters.bootstrap.default_chunking_engine import (
                create_default_document_splitter,
            )

            splitter = create_default_document_splitter()
        self._loader = loader
        self._splitter = splitter
        self._trace_writer = trace_writer
        self._event_bus = event_bus

    def bind_trace_writer(self, trace_writer: RunTraceWriter) -> None:
        """Late-bind trace writer when created after service construction (RuntimeContext.build)."""
        self._trace_writer = trace_writer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest_attachments_for_session(
        self,
        attachments: Sequence[AttachmentRef],
        *,
        session_id: str,
        user_id: str,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> List[IngestionResult]:
        """
        Ingest all provided attachments in the context of a specific session.

        The session/user/tenant/workspace identifiers are injected as metadata,
        so that RAG retrieval can later filter documents appropriately.
        """

        if self._embedding_manager is None or self._vectorstore_manager is None:
            raise ValueError(
                "Attachment ingestion requires embedding_manager and vectorstore_manager. "
                "Provide them in RuntimeConfig or pass a custom ingestion_service."
            )

        results: List[IngestionResult] = []

        for attachment in attachments:
            try:
                result = await self._ingest_single_attachment(
                    attachment=attachment,
                    session_id=session_id,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    run_id=run_id,
                )
            except (FileNotFoundError, ValueError, OSError) as exc:
                record_ingestion_failed(
                    self._event_bus,
                    attachment_id=attachment.id,
                    session_id=session_id,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    run_id=run_id,
                    error=exc,
                )
                result = IngestionResult(
                    attachment_id=attachment.id,
                    attachment_type=attachment.type,
                    num_chunks=0,
                    vector_ids=[],
                    metadata={
                        "reason": "ingestion_failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
            results.append(result)

        return results

    
    async def search_session_attachments(
        self,
        *,
        query: str,
        session_id: str,
        user_id: str,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        top_k: int = 6,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Retrieval over session attachments indexed by this ingestion service.

        Contract:
          - Uses ingestion embedding_manager to embed the user query
          - Uses ingestion vectorstore_manager.query(query_embedding, top_k, metadata_filter=...)
        """
        q = (query or "").strip()
        if not q:
            return {
                "used": False,
                "hits": [],
                "scores": [],
                "debug": {"reason": "empty_query"},
            }

        if self._vectorstore_manager is None:
            return {
                "used": False,
                "hits": [],
                "scores": [],
                "debug": {"reason": "vectorstore_manager_not_configured"},
            }

        if self._embedding_manager is None:
            return {
                "used": False,
                "hits": [],
                "scores": [],
                "debug": {"reason": "embedding_manager_not_configured"},
            }

        where: Dict[str, Any] = {
            "session_id": session_id,
            "user_id": user_id,
        }
        if tenant_id is not None:
            where["tenant_id"] = tenant_id
        if workspace_id is not None:
            where["workspace_id"] = workspace_id

        metadata_filter = MetadataFilter(conditions=where) if where else None

        # 1) Embed the query using ingestion embedder.
        # Expected EmbeddingManager API in Intergrax: embed_one / embed_query / embed
        # Use the same call you use in ingestion indexing for chunks, but for single query string.
        query_emb = self._embedding_manager.embed_one(q)        
        if inspect.iscoroutine(query_emb):
            query_emb = await query_emb

        # 2) Vector search in ingestion vectorstore with strict session filters.
        raw = self._vectorstore_manager.query(
            query_embedding=query_emb,
            top_k=int(top_k),
            metadata_filter=metadata_filter,
            include_embeddings=False,
        )
        
        hits: List[RetrievedChunk] = []
        scores: List[float] = []

        for i, raw_hit in enumerate(raw):
            if not isinstance(raw_hit, VectorStoreHit):
                return {
                    "used": False,
                    "hits": [],
                    "scores": [],
                    "debug": {"reason": "unexpected_vectorstore_result_type"},
                }

            text = (raw_hit.content or "").strip()
            if not text:
                continue

            md = dict(raw_hit.metadata or {})
            scv = float(raw_hit.similarity_score)

            md["vector_id"] = raw_hit.id
            chunk_id = str(raw_hit.id) if raw_hit.id is not None else f"{session_id}:{i}"

            hits.append(RetrievedChunk(chunk_id, text, md, scv))
            scores.append(scv)

        # Optional threshold filtering.
        if score_threshold is not None and hits:
            thr = float(score_threshold)
            filt_hits: List[RetrievedChunk] = []
            filt_scores: List[float] = []
            for h, sc in zip(hits, scores):
                if sc >= thr:
                    filt_hits.append(h)
                    filt_scores.append(sc)
            hits = filt_hits
            scores = filt_scores

        used = bool(hits)

        return {
            "used": used,
            "hits": hits,
            "scores": scores,
            "debug": {
                "used": used,
                "hits_count": len(hits),
                "top_k": int(top_k),
                "score_threshold": score_threshold,
                "where": where,
                "provider": type(self._vectorstore_manager).__name__,
            },
        }


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ingest_single_attachment(
        self,
        attachment: AttachmentRef,
        *,
        session_id: str,
        user_id: str,
        tenant_id: Optional[str],
        workspace_id: Optional[str],
        run_id: Optional[str] = None,
    ) -> IngestionResult:
        """
        End-to-end ingestion pipeline for a single AttachmentRef.
        """
        # 1) Resolve AttachmentRef → Path (or raise FileNotFoundError/ValueError)
        path: Path = await self._resolver.resolve_to_path(attachment)

        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id is required for attachment ingestion")

        # 2) Build base metadata that we want on every chunk
        base_metadata: Dict[str, Any] = {
            "attachment_id": attachment.id,
            "attachment_type": attachment.type,
            "session_id": session_id,
            "user_id": user_id,
            "workspace_id": workspace_id,
        }
        if attachment.metadata:
            base_metadata.update(attachment.metadata)

        # 3) Use IntergraxDocumentsLoader.load_document(...) for a single file
        def _metadata_callback(doc: KnowledgeDocument, p: str) -> Dict[str, Any]:
            """
            Custom metadata callback for the IntergraxDocumentsLoader.

            It receives each loaded KnowledgeDocument and its Path, and returns a dict
            merged into doc.metadata. We always inject our base_metadata, but
            we do not override keys that the loader already set (unless they
            are absent).
            """
            merged = dict(base_metadata)
            # Optionally, we could inspect doc.metadata here and adjust.
            return merged

        native_docs = self._loader.load_document(
            str(path),
            tenant_id=tenant_id,
            namespace=workspace_id,
            use_default_metadata=True,
            call_custom_metadata=_metadata_callback,
        )

        if native_docs and self._trace_writer is not None:
            from intergrax.rag.document_loaders.pipeline.parser_pipeline import TRACE_METADATA_KEY
            from intergrax.runtime.nexus.tracing.parser_trace_span import maybe_append_parser_trace

            first_meta = dict(native_docs[0].metadata)
            trace = dict(first_meta.get(TRACE_METADATA_KEY) or {})
            if trace:
                maybe_append_parser_trace(
                    self._trace_writer,
                    run_id=run_id,
                    source=str(path),
                    trace=trace,
                    session_id=session_id,
                    user_id=user_id,
                    tenant_id=tenant_id or "",
                )

        if not native_docs:
            return IngestionResult(
                attachment_id=attachment.id,
                attachment_type=attachment.type,
                num_chunks=0,
                vector_ids=[],
                metadata={
                    "reason": "no_documents_loaded",
                    "source_path": str(path),
                },
            )

        # 4) Split into chunks via IntergraxDocumentsSplitter
        native_chunks = self._splitter.split_documents(native_docs)
        chunks: List[Document] = [
            to_legacy_rag_document(chunk) for chunk in native_chunks
        ]

        if not chunks:
            return IngestionResult(
                attachment_id=attachment.id,
                attachment_type=attachment.type,
                num_chunks=0,
                vector_ids=[],
                metadata={
                    "reason": "no_chunks_generated",
                    "source_path": str(path),
                },
            )

        # 5) Embed chunks and store in vectorstore
        #
        # The IntergraxEmbeddingManager / IntergraxVectorstoreManager in your
        # project are currently synchronous. However, to keep this runtime
        # future-proof, we support both sync and async interfaces.
        #
        # Pattern:
        #   result = func(...)
        #   if inspect.iscoroutine(result): await it
        #   else: use it directly

        # 5a) Embeddings
        try:
            embed_result = self._embedding_manager.embed_documents(chunks)

            if inspect.iscoroutine(embed_result):
                embed_result = await embed_result

            aligned_docs = embed_result.documents
            embeddings = embed_result.embeddings

        except AttributeError:
            # Fallback: manager exposes only embed_texts(texts)

            texts = [c.page_content for c in chunks]

            embed_result = self._embedding_manager.embed_texts(texts)

            if inspect.iscoroutine(embed_result):
                embeddings = await embed_result
            else:
                embeddings = embed_result

            aligned_docs = chunks

        # 5b) Enrich metadata on documents with base_metadata
        #
        # This ensures that later retrieval can filter by session/tenant/user/etc.
        for d in aligned_docs:
            d.metadata = {**(d.metadata or {}), **base_metadata}

        # 5c) Generate stable IDs for each stored chunk
        ids = [f"{attachment.id}-{i}" for i in range(len(aligned_docs))]

        # 5d) Store in vectorstore using the current IntergraxVectorstoreManager API.
        #
        # We assume a signature similar to:
        #   add_documents(
        #       documents: Sequence[Document],
        #       embeddings: Optional[Any] = None,
        #       ids: Optional[Sequence[str]] = None,
        #       base_metadata: Optional[Dict[str, Any]] = None,
        #       ...
        #   )
        add_result = self._vectorstore_manager.add_documents(
            documents=aligned_docs,
            embeddings=embeddings,
            ids=ids,
            base_metadata=base_metadata,
        )

        if inspect.iscoroutine(add_result):
            stored_ids = await add_result
        else:
            stored_ids = add_result

        # Normalize stored_ids: if the manager returns None, fall back to local ids
        if stored_ids is None:
            vector_ids = ids
        else:
            vector_ids = list(stored_ids)

        parser_trace: Dict[str, Any] = {}
        parser_id: str | None = None
        if aligned_docs:
            first_meta = aligned_docs[0].metadata or {}
            from intergrax.rag.document_loaders.pipeline.parser_pipeline import TRACE_METADATA_KEY

            if TRACE_METADATA_KEY in first_meta:
                parser_trace = dict(first_meta.get(TRACE_METADATA_KEY) or {})
                parser_id = parser_trace.get("parser_id") or first_meta.get("integration_parser_id")

        return IngestionResult(
            attachment_id=attachment.id,
            attachment_type=attachment.type,
            num_chunks=len(aligned_docs),
            vector_ids=vector_ids,
            metadata={
                "source_path": str(path),
                "session_id": session_id,
                "user_id": user_id,
                "tenant_id": tenant_id,
                "workspace_id": workspace_id,
                "integration_parser_id": parser_id,
                "integration_parser_trace": parser_trace,
            },
        )

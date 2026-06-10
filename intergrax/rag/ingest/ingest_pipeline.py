# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Configurable ingest pipeline — loader, chunking strategy, optional contextual enrich."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from langchain_core.documents import Document

from intergrax.rag.contextual.chunk_enricher import ContextualChunkEnricher
from intergrax.rag.document_loaders.contracts.base_document_loader import BaseDocumentsLoader
from intergrax.rag.document_loaders.pipeline.parser_pipeline import TRACE_METADATA_KEY
from intergrax.rag.document_splitters.contracts.base_documents_splitter import BaseDocumentsSplitter
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.graph.contracts.graph_store import GraphStore
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.graph.indexer.graph_indexer_factory import resolve_graph_indexer
from intergrax.rag.ingest.ingest_policy import sync_ingest_allowed
from intergrax.rag.indexing.indexing_manager import IndexingManager
from intergrax.rag.indexing.strategies.dual_index_strategy import DualIndexStrategy
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.tracking.rag_spans import rag_span
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager


@dataclass(frozen=True)
class IngestRequest:
    source_path: str
    base_metadata: Dict[str, Any] = field(default_factory=dict)
    chunking_strategy_id: Optional[str] = None


@dataclass
class IngestResult:
    used: bool
    reason: str
    file_size_bytes: int = 0
    async_job_recommended: bool = False
    num_chunks: int = 0
    vector_ids: List[str] = field(default_factory=list)
    parser_id: Optional[str] = None
    parser_trace: Dict[str, Any] = field(default_factory=dict)


class IngestPipeline:
    def __init__(
        self,
        *,
        loader: BaseDocumentsLoader,
        splitter: BaseDocumentsSplitter,
        embedding_manager: BaseEmbeddingManager,
        vectorstore: BaseVectorstoreManager,
        toc_vectorstore: Optional[BaseVectorstoreManager] = None,
        profile: Optional[RagProfile] = None,
        contextual_enricher: Optional[ContextualChunkEnricher] = None,
        graph_store: Optional[GraphStore] = None,
        llm_for_graph: Optional[LLMAdapter] = None,
        metadata_callback: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self._loader = loader
        self._splitter = splitter
        self._embedding_manager = embedding_manager
        self._vectorstore = vectorstore
        self._toc_vectorstore = toc_vectorstore
        self._profile = profile or RagProfile()
        self._contextual = contextual_enricher
        self._graph_store = graph_store
        self._llm_for_graph = llm_for_graph
        self._metadata_callback = metadata_callback

    def run(self, request: IngestRequest) -> IngestResult:
        path = Path(request.source_path)
        with rag_span(
            "rag.ingest",
            attributes={
                "rag.ingest.source": str(path),
                "rag.ingest.chunking_strategy": request.chunking_strategy_id
                or self._profile.chunking_strategy_id,
            },
        ):
            if not path.exists():
                return IngestResult(used=False, reason="source_not_found")

            allowed, size_reason, file_size = sync_ingest_allowed(path=path, profile=self._profile)
            if not allowed:
                return IngestResult(
                    used=False,
                    reason=size_reason,
                    file_size_bytes=file_size,
                    async_job_recommended=True,
                )

            base_metadata = dict(request.base_metadata)
            if self._profile.embedding_model_version:
                base_metadata["embedding_model_version"] = self._profile.embedding_model_version

            def _cb(doc: Any, source: str) -> Dict[str, Any]:
                if self._metadata_callback is not None:
                    return self._metadata_callback(doc, source)
                return dict(base_metadata)

            with rag_span("rag.ingest.load", attributes={"rag.ingest.source": str(path)}):
                docs = self._loader.load_document(
                    str(path),
                    use_default_metadata=True,
                    call_custom_metadata=_cb,
                )
            if not docs:
                return IngestResult(used=False, reason="no_documents_loaded")

            parser_trace: Dict[str, Any] = {}
            parser_id: Optional[str] = None
            first_meta = docs[0].metadata or {}
            if TRACE_METADATA_KEY in first_meta:
                parser_trace = dict(first_meta.get(TRACE_METADATA_KEY) or {})
                parser_id = parser_trace.get("parser_id") or first_meta.get("integration_parser_id")

            strategy_id = request.chunking_strategy_id or self._profile.chunking_strategy_id
            with rag_span(
                "rag.ingest.chunk",
                attributes={"rag.ingest.chunking_strategy": strategy_id},
            ):
                try:
                    chunks = self._splitter.split_documents(docs, strategy_id=strategy_id)
                except TypeError:
                    chunks = self._splitter.split_documents(docs)
            if not chunks:
                return IngestResult(
                    used=False,
                    reason="no_chunks_generated",
                    parser_id=parser_id,
                    parser_trace=parser_trace,
                )

            chunk_list: Sequence[Document] = list(chunks)
            if self._profile.contextual_enrich == "on" and self._contextual is not None:
                chunk_list = self._contextual.enrich(docs, chunk_list)

            aligned_docs = list(chunk_list)
            for doc in aligned_docs:
                doc.metadata = {**(doc.metadata or {}), **base_metadata}

            ids = [f"ingest-{path.stem}-{i}" for i in range(len(aligned_docs))]

            with rag_span(
                "rag.ingest.index",
                attributes={
                    "rag.ingest.num_chunks": len(aligned_docs),
                    "rag.ingest.dual_index": self._uses_dual_index(),
                },
            ):
                if self._uses_dual_index():
                    assert self._toc_vectorstore is not None
                    IndexingManager(
                        embed_manager=self._embedding_manager,
                        vectorstore=self._vectorstore,
                        strategy=DualIndexStrategy(toc_vectorstore=self._toc_vectorstore),
                    ).index_documents(aligned_docs)
                    vector_ids = ids
                else:
                    try:
                        embed_result = self._embedding_manager.embed_documents(aligned_docs)
                        aligned_docs = list(embed_result.documents)
                        embeddings = embed_result.embeddings
                    except AttributeError:
                        texts = [c.page_content for c in aligned_docs]
                        embeddings = self._embedding_manager.embed_texts(texts)

                    stored_ids = self._vectorstore.add_documents(
                        documents=aligned_docs,
                        embeddings=embeddings,
                        ids=ids,
                        base_metadata=base_metadata,
                    )
                    vector_ids = list(stored_ids) if stored_ids is not None else ids

            if self._graph_store is not None and self._profile.graph_rag_enabled:
                with rag_span("rag.ingest.graph_index"):
                    indexer = resolve_graph_indexer(
                        self._graph_store,
                        self._profile,
                        llm=self._llm_for_graph,
                    )
                    indexer.index_documents(aligned_docs, chunk_ids=vector_ids)

            return IngestResult(
                used=True,
                reason="ok",
                num_chunks=len(aligned_docs),
                vector_ids=vector_ids,
                parser_id=parser_id,
                parser_trace=parser_trace,
            )

    def _uses_dual_index(self) -> bool:
        return self._profile.uses_hierarchical_index() and self._toc_vectorstore is not None

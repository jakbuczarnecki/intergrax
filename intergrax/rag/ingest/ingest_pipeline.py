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
from intergrax.rag.profiles.rag_profile import RagProfile
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
        profile: Optional[RagProfile] = None,
        contextual_enricher: Optional[ContextualChunkEnricher] = None,
        metadata_callback: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self._loader = loader
        self._splitter = splitter
        self._embedding_manager = embedding_manager
        self._vectorstore = vectorstore
        self._profile = profile or RagProfile()
        self._contextual = contextual_enricher
        self._metadata_callback = metadata_callback

    def run(self, request: IngestRequest) -> IngestResult:
        path = Path(request.source_path)
        if not path.exists():
            return IngestResult(used=False, reason="source_not_found")

        base_metadata = dict(request.base_metadata)
        if self._profile.embedding_model_version:
            base_metadata["embedding_model_version"] = self._profile.embedding_model_version

        def _cb(doc: Any, source: str) -> Dict[str, Any]:
            if self._metadata_callback is not None:
                return self._metadata_callback(doc, source)
            return dict(base_metadata)

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

        try:
            embed_result = self._embedding_manager.embed_documents(chunk_list)
            aligned_docs = embed_result.documents
            embeddings = embed_result.embeddings
        except AttributeError:
            texts = [c.page_content for c in chunk_list]
            embeddings = self._embedding_manager.embed_texts(texts)
            aligned_docs = list(chunk_list)

        for doc in aligned_docs:
            doc.metadata = {**(doc.metadata or {}), **base_metadata}

        ids = [f"ingest-{path.stem}-{i}" for i in range(len(aligned_docs))]
        stored_ids = self._vectorstore.add_documents(
            documents=aligned_docs,
            embeddings=embeddings,
            ids=ids,
            base_metadata=base_metadata,
        )
        vector_ids = list(stored_ids) if stored_ids is not None else ids

        return IngestResult(
            used=True,
            reason="ok",
            num_chunks=len(aligned_docs),
            vector_ids=vector_ids,
            parser_id=parser_id,
            parser_trace=parser_trace,
        )

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Attachment ingestion pipeline for Drop-In Knowledge Mode.

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

from intergrax.llm.messages import AttachmentRef
from intergrax.rag.documents_loader import DocumentsLoader
from intergrax.rag.documents_splitter import DocumentsSplitter
from intergrax.rag.embedding_manager import EmbeddingManager
from intergrax.rag.vectorstore_manager import VectorstoreManager

from .attachments import AttachmentResolver


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
    High-level ingestion service for Drop-In Knowledge Mode.

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

    It is intended to be called from DropInKnowledgeRuntime or other
    orchestration layers when new attachments are added to a session.
    """

    def __init__(
        self,
        *,
        resolver: AttachmentResolver,
        embedding_manager: EmbeddingManager,
        vectorstore_manager: VectorstoreManager,
        loader: Optional[DocumentsLoader] = None,
        splitter: Optional[DocumentsSplitter] = None,
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
        self._loader = loader or DocumentsLoader(verbose=False)
        self._splitter = splitter or DocumentsSplitter(
            verbose=False,
            default_chunk_size=1000,
            default_chunk_overlap=100,
        )

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
    ) -> List[IngestionResult]:
        """
        Ingest all provided attachments in the context of a specific session.

        The session/user/tenant/workspace identifiers are injected as metadata,
        so that RAG retrieval can later filter documents appropriately.
        """
        results: List[IngestionResult] = []

        for attachment in attachments:
            result = await self._ingest_single_attachment(
                attachment=attachment,
                session_id=session_id,
                user_id=user_id,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
            results.append(result)

        return results

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
    ) -> IngestionResult:
        """
        End-to-end ingestion pipeline for a single AttachmentRef.
        """
        # 1) Resolve AttachmentRef → Path (or raise FileNotFoundError/ValueError)
        path: Path = await self._resolver.resolve_to_path(attachment)

        # 2) Build base metadata that we want on every chunk
        base_metadata: Dict[str, Any] = {
            "attachment_id": attachment.id,
            "attachment_type": attachment.type,
            "session_id": session_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
        }
        if attachment.metadata:
            base_metadata.update(attachment.metadata)

        # 3) Use IntergraxDocumentsLoader.load_document(...) for a single file
        def _metadata_callback(doc: Document, p: Path) -> Dict[str, Any]:
            """
            Custom metadata callback for the IntergraxDocumentsLoader.

            It receives each loaded Document and its Path, and returns a dict
            merged into doc.metadata. We always inject our base_metadata, but
            we do not override keys that the loader already set (unless they
            are absent).
            """
            merged = dict(base_metadata)
            # Optionally, we could inspect doc.metadata here and adjust.
            return merged

        docs: List[Document] = self._loader.load_document(
            str(path),
            use_default_metadata=True,
            call_custom_metadata=_metadata_callback,
        )

        if not docs:
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
        chunks: List[Document] = self._splitter.split_documents(docs)

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
            # Preferred path: the manager exposes embed_documents(chunks)
            embed_result = self._embedding_manager.embed_documents(chunks)

            if inspect.iscoroutine(embed_result):
                embed_result = await embed_result

            # Normalize result: either (embeddings, docs) or embeddings-only
            if isinstance(embed_result, tuple) and len(embed_result) == 2:
                embeddings, aligned_docs = embed_result
            else:
                embeddings = embed_result
                aligned_docs = chunks

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
            },
        )

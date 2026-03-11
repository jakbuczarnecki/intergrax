# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Dict, Set
import math

from langchain_core.documents import Document

from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey
from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from intergrax.rag.indexing.contracts.index_strategy import IndexStrategy


class DualIndexStrategy(IndexStrategy):
    """
    Dual index strategy.

    Builds two indexes:

        CHUNKS → full chunk index
        TOC    → section index derived from chunk metadata
    """

    def __init__(
        self,
        *,
        toc_vectorstore: VectorstoreManager,
        batch_size: int = 512,
    ):
        self.toc_vectorstore = toc_vectorstore
        self.batch_size = batch_size

    def build_index(
        self,
        *,
        documents: List[Document],
        embed_manager: EmbeddingManager,
        vectorstore: VectorstoreManager,
    ) -> None:

        if not documents:
            return

        # -----------------------------
        # Main CHUNK index
        # -----------------------------

        embeddings, aligned_docs = embed_manager.embed_documents(documents)

        self._insert_batches(
            vectorstore,
            aligned_docs,
            embeddings,
        )

        # -----------------------------
        # Build TOC index from SECTION metadata
        # -----------------------------

        sections: Dict[str, Dict] = {}

        for doc in documents:

            metadata = doc.metadata or {}

            section = metadata.get(ChunkMetadataKey.SECTION)

            if not section:
                continue

            if section not in sections:

                sections[section] = {
                    "text": section,
                    "metadata": metadata,
                }

        if not sections:
            return

        toc_docs: List[Document] = []

        for section_name, data in sections.items():

            toc_docs.append(
                Document(
                    page_content=section_name,
                    metadata=data["metadata"],
                )
            )

        embeddings, aligned_docs = embed_manager.embed_documents(toc_docs)

        self._insert_batches(
            self.toc_vectorstore,
            aligned_docs,
            embeddings,
        )

    def _insert_batches(
        self,
        vectorstore: VectorstoreManager,
        documents: List[Document],
        embeddings,
    ) -> None:

        n = len(documents)

        for i in range(0, n, self.batch_size):

            j = min(i + self.batch_size, n)

            vectorstore.add_documents(
                documents=documents[i:j],
                embeddings=embeddings[i:j],
                batch_size=self.batch_size,
            )
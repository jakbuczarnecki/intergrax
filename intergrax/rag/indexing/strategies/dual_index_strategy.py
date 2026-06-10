# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Dict

from langchain_core.documents import Document

from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
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
        toc_vectorstore: BaseVectorstoreManager,
        batch_size: int = 512,
    ):
        self.toc_vectorstore = toc_vectorstore
        self.batch_size = batch_size

    def build_index(
        self,
        *,
        documents: List[Document],
        embed_manager: BaseEmbeddingManager,
        vectorstore: BaseVectorstoreManager,
    ) -> None:

        if not documents:
            return

        # -----------------------------
        # Main CHUNK index
        # -----------------------------

        result = embed_manager.embed_documents(documents)

        self._insert_batches(
            vectorstore,
            result.documents,
            result.embeddings,
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
            toc_meta = dict(data["metadata"])
            parent_id = toc_meta.get(ChunkMetadataKey.PARENT_CHUNK_ID) or section_name
            toc_meta[ChunkMetadataKey.PARENT_CHUNK_ID] = str(parent_id)
            toc_meta[ChunkMetadataKey.SECTION] = section_name
            toc_docs.append(
                Document(
                    page_content=section_name,
                    metadata=toc_meta,
                )
            )

        result = embed_manager.embed_documents(toc_docs)

        self._insert_batches(
            self.toc_vectorstore,
            result.documents,
            result.embeddings,
        )

    def _insert_batches(
        self,
        vectorstore: BaseVectorstoreManager,
        documents: List[Document],
        embeddings,
    ) -> None:

        n = len(documents)

        for i in range(0, n, self.batch_size):

            j = min(i + self.batch_size, n)

            vectorstore.add_documents(
                documents=documents[i:j],
                embeddings=embeddings[i:j],
            )
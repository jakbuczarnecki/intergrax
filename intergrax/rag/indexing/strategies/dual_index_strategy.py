# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_splitters.chunk_document import build_derived_chunk

from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreRecord,
)
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
        documents: Sequence[KnowledgeDocument],
        embed_manager: BaseEmbeddingManager,
        vectorstore: BaseVectorstoreManager,
    ) -> None:

        if not documents:
            return

        # -----------------------------
        # Main CHUNK index
        # -----------------------------

        main_result = embed_manager.embed_documents(documents)

        self._insert_batches(
            vectorstore,
            main_result.documents,
            main_result.embeddings,
        )

        # -----------------------------
        # Build TOC index from SECTION metadata
        # -----------------------------

        sections: dict[tuple[str, str | None, str, str], KnowledgeDocument] = {}

        for doc in documents:

            section = doc.metadata.get(ChunkMetadataKey.SECTION.value)

            if not isinstance(section, str) or not section.strip():
                continue

            group_key = (
                doc.scope.tenant_id,
                doc.scope.namespace,
                doc.identity.root_document_id,
                section,
            )
            sections.setdefault(group_key, doc)

        if not sections:
            return

        toc_docs: list[KnowledgeDocument] = []

        for (
            tenant_id,
            namespace,
            root_document_id,
            section_name,
        ), parent_document in sections.items():
            toc_docs.append(
                build_derived_chunk(
                    parent_document,
                    content=section_name,
                    strategy_id=(
                        f"toc:{tenant_id!r}:{namespace!r}:{root_document_id!r}"
                    ),
                    chunk_index=0,
                    metadata_updates={
                        ChunkMetadataKey.PARENT_CHUNK_ID.value: (
                            parent_document.identity.document_id
                        ),
                        ChunkMetadataKey.SECTION.value: section_name,
                    },
                )
            )

        toc_result = embed_manager.embed_documents(toc_docs)

        self._insert_batches(
            self.toc_vectorstore,
            toc_result.documents,
            toc_result.embeddings,
        )

    def _insert_batches(
        self,
        vectorstore: BaseVectorstoreManager,
        documents: Sequence[KnowledgeDocument],
        embeddings: NDArray[np.float32],
    ) -> None:

        n = len(documents)

        for i in range(0, n, self.batch_size):

            j = min(i + self.batch_size, n)

            batch_documents = documents[i:j]
            records = [
                VectorStoreRecord(
                    document=document,
                    embedding=embeddings[index],
                    vector_id=document.identity.document_id,
                )
                for index, document in enumerate(batch_documents, start=i)
            ]

            vectorstore.add_records(records)
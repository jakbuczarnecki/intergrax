# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence, List

import numpy as np
from numpy.typing import NDArray

from langchain_core.documents import Document

from intergrax.rag.embedding.contracts.embedding_metadata_key import EmbeddingMetadataKey
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.embedding.engine.embedding_engine import EmbeddingEngine


class EmbeddingPipeline:
    """
    Pipeline responsible for embedding generation.

    Responsibilities
    ----------------
    - accept texts or documents
    - delegate embedding computation to EmbeddingEngine
    - attach vectors to document metadata when required
    """

    def __init__(
        self,
        engine: EmbeddingEngine,
        *,
        provider_id: str,
    ) -> None:
        self._engine = engine
        self._provider_id = provider_id


    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float32]:
        """
        Generate embeddings for a sequence of texts.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        return self._engine.embed(
            texts,
            provider_id=self._provider_id,
        )


    def embed_one(
        self,
        text: str,
    ) -> NDArray[np.float32]:
        """
        Generate embedding for a single text.
        """
        vecs = self.embed_texts([text])

        if vecs.size == 0:
            return vecs

        return vecs[0:1]


    def embed_documents(
        self,
        documents: Sequence[Document],
    ) -> EmbeddingResult:
        """
        Generate embeddings for documents and attach them to metadata.
        """

        if not documents:
            return EmbeddingResult(
                documents=[],
                embeddings=[],
            )

        texts: List[str] = [doc.page_content for doc in documents]

        embeddings = self.embed_texts(texts)

        enriched_documents: List[Document] = []

        for document, vector in zip(documents, embeddings):

            metadata = dict(document.metadata or {})
            metadata[EmbeddingMetadataKey.VECTOR] = vector

            enriched_documents.append(
                Document(
                    page_content=document.page_content,
                    metadata=metadata,
                )
            )

        return EmbeddingResult(
            documents=enriched_documents,
            embeddings=embeddings,
        )
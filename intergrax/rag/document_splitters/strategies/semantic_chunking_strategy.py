# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence, List

import numpy as np
from langchain_core.documents import Document

from intergrax.rag.document_splitters.contracts.base_chunking_strategy import BaseChunkingStrategy
from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey
from intergrax.rag.embedding.embedding_manager import EmbeddingManager


class SemanticChunkingStrategy(BaseChunkingStrategy):
    """
    Semantic chunking strategy based on embedding similarity.

    Documents are segmented using sentence embeddings and cosine similarity
    to detect semantic boundaries.
    """

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        similarity_threshold: float = 0.75,
    ) -> None:
        self._embedding_manager = embedding_manager
        self._threshold = similarity_threshold

    @classmethod
    def strategy_id(cls) -> str:
        return "semantic"

    def _cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (norm_a * norm_b)

    def chunk(
        self,
        documents: Sequence[Document],
    ) -> Sequence[Document]:

        chunks: List[Document] = []

        for doc in documents:

            text = doc.page_content
            metadata = dict(doc.metadata)

            sentences = [s.strip() for s in text.split(".") if s.strip()]

            if not sentences:
                continue

            embeddings = self._embedding_manager.embed_texts(sentences)

            current_chunk: List[str] = [sentences[0]]
            chunk_index = 0

            for i in range(1, len(sentences)):

                sim = self._cosine_similarity(
                    embeddings[i - 1],
                    embeddings[i],
                )

                if sim < self._threshold:

                    chunk_text = ". ".join(current_chunk)

                    chunk_metadata = dict(metadata)
                    chunk_metadata[ChunkMetadataKey.CHUNK_INDEX] = chunk_index
                    chunk_metadata[ChunkMetadataKey.CHUNK_STRATEGY] = self.strategy_id()
                    chunk_metadata[ChunkMetadataKey.CHUNK_SIZE] = len(chunk_text)

                    chunks.append(
                        Document(
                            page_content=chunk_text,
                            metadata=chunk_metadata,
                        )
                    )

                    current_chunk = []
                    chunk_index += 1

                current_chunk.append(sentences[i])

            if current_chunk:

                chunk_text = ". ".join(current_chunk)

                chunk_metadata = dict(metadata)
                chunk_metadata[ChunkMetadataKey.CHUNK_INDEX] = chunk_index
                chunk_metadata[ChunkMetadataKey.CHUNK_STRATEGY] = self.strategy_id()
                chunk_metadata[ChunkMetadataKey.CHUNK_SIZE] = len(chunk_text)

                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=chunk_metadata,
                    )
                )

        return chunks
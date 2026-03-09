# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from langchain_core.documents import Document

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.embedding_pipeline import EmbeddingPipeline

class EmbeddingManager(BaseEmbeddingManager):
    """
    Entry point for embedding generation.

    Responsibilities
    ----------------
    - accept texts or documents
    - delegate embedding execution to EmbeddingPipeline

    This class intentionally contains no embedding logic.
    All embedding behaviour is implemented by providers
    registered in EmbeddingProviderRegistry and executed
    by EmbeddingEngine through EmbeddingPipeline.
    """

    def __init__(
        self,
        *,
        pipeline: EmbeddingPipeline,
    ) -> None:
        self._pipeline = pipeline


    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float32]:
        """
        Generate embeddings for raw texts.
        """
        return self._pipeline.embed_texts(texts)


    def embed_one(
        self,
        text: str,
    ) -> NDArray[np.float32]:
        """
        Generate embedding for a single text.
        """
        return self._pipeline.embed_one(text)


    def embed_documents(
        self,
        documents: Sequence[Document],
    ) -> Sequence[Document]:
        """
        Generate embeddings for LangChain Document objects.
        """
        return self._pipeline.embed_documents(documents)
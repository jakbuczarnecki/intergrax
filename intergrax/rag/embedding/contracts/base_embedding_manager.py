# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult


class BaseEmbeddingManager(ABC):
    
    @abstractmethod
    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float32]:
        """
        Embed raw text strings.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_one(
        self,
        text: str,
    ) -> NDArray[np.float32]:
        """
        Embed one raw text string.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_documents(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> EmbeddingResult:
        """
        Embed native documents without mutating them.

        The result preserves input order and keeps documents separate from
        their embedding vectors.
        """
        raise NotImplementedError
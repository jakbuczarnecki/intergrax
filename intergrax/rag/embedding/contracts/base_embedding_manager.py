# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence

from langchain_core.documents import Document
import numpy as np
from numpy.typing import NDArray


class BaseEmbeddingManager(ABC):
    
    @abstractmethod
    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float32]:
        """
        Generate embeddings for raw texts.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_one(
        self,
        text: str,
    ) -> NDArray[np.float32]:
        """
        Generate embedding for a single text.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_documents(
        self,
        documents: Sequence[Document],
    ) -> Sequence[Document]:
        """
        Generate embeddings for LangChain Document objects.
        """
        raise NotImplementedError
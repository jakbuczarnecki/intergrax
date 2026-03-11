# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from langchain_openai import OpenAIEmbeddings

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider.
    """

    DEFAULT_MODEL = "text-embedding-3-small"    
    ENV_MODEL = "INTERGRAX_DEFAULT_OPENAI_EMBED_MODEL"

    def __init__(
            self, 
            model_name: Optional[str] = None
    ) -> None:
        
        env_model = os.getenv(self.ENV_MODEL)
        self._model_name = model_name or env_model or self.DEFAULT_MODEL        
        self._model: Optional[OpenAIEmbeddings] = None
        self._dim: Optional[int] = None
        

    def provider_name(self) -> str:
        return "openai"
    
    def _ensure_model(self):
        if self._model is None:
            self._model = OpenAIEmbeddings(model=self._model_name)

    def _resolve_dim(self):

        self._ensure_model()

        if self._dim is None:
            # probe embedding dimension
            test_vec = self._model.embed_query("probe-dimension")
            self._dim = int(len(test_vec)) if test_vec else 0

    def dimension(self) -> int:
        self._resolve_dim()
        return self._dim

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        
        self._ensure_model()

        vecs = self._model.embed_documents(list(texts))

        arr = np.asarray(vecs, dtype=np.float32)

        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)

        return arr
# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from langchain_openai import OpenAIEmbeddings

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider.
    """

    def __init__(self, model_name: str) -> None:
        self._model = OpenAIEmbeddings(model=model_name)

        # probe embedding dimension
        test_vec = self._model.embed_query("probe-dimension")
        self._dim = int(len(test_vec)) if test_vec else 0

    def provider_name(self) -> str:
        return "openai"

    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:

        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        vecs = self._model.embed_documents(list(texts))

        arr = np.asarray(vecs, dtype=np.float32)

        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)

        return arr
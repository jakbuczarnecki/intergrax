# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from langchain_ollama import OllamaEmbeddings

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Ollama embedding provider.
    """

    DEFAULT_MODEL = "rjmalagon/gte-qwen2-1.5b-instruct-embed-f16:latest"
    ENV_MODEL = "INTERGRAX_DEFAULT_OLLAMA_EMBED_MODEL"

    def __init__(
        self,
        model_name: Optional[str] = None
    ) -> None:

        env_model = os.getenv(self.ENV_MODEL)
        resolved_model = model_name or env_model or self.DEFAULT_MODEL

        # store configuration only
        self._model_name = resolved_model
        self._model: Optional[OllamaEmbeddings] = None

        self._dim: Optional[int] = None

    def provider_name(self) -> str:
        return "ollama"

    def _ensure_model(self) -> None:

        if self._model is None:
            self._model = OllamaEmbeddings(model=self._model_name)

    def _resolve_dim(self) -> None:

        if self._dim is None:

            self._ensure_model()

            test_vec = self._model.embed_query("probe-dimension")
            self._dim = int(len(test_vec)) if test_vec else 0

    def dimension(self) -> int:

        self._resolve_dim()
        return self._dim

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:

        if not texts:
            self._resolve_dim()
            return np.empty((0, self._dim), dtype=np.float32)

        self._ensure_model()

        vecs = self._model.embed_documents(list(texts))

        arr = np.asarray(vecs, dtype=np.float32)

        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)

        return arr
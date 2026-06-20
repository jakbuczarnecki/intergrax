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


class VllmEmbeddingProvider(EmbeddingProvider):
    """
    vLLM embedding provider (OpenAI-compatible ``/v1/embeddings``).

    Requires a vLLM server started with a pooling/embedding model, e.g.
    ``vllm serve BAAI/bge-small-en-v1.5 --runner pooling``.
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    ENV_MODEL = "INTERGRAX_DEFAULT_VLLM_EMBED_MODEL"
    ENV_BASE_URL = "INTERGRAX_DEFAULT_VLLM_EMBED_BASE_URL"
    ENV_FALLBACK_BASE_URL = "INTERGRAX_DEFAULT_VLLM_BASE_URL"
    ENV_API_KEY = "VLLM_API_KEY"
    DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

    def __init__(self, model_name: Optional[str] = None) -> None:
        env_model = os.getenv(self.ENV_MODEL)
        self._model_name = model_name or env_model or self.DEFAULT_MODEL
        self._model: Optional[OpenAIEmbeddings] = None
        self._dim: Optional[int] = None

    def provider_name(self) -> str:
        return "vllm"

    def _resolve_base_url(self) -> str:
        raw = (
            os.getenv(self.ENV_BASE_URL)
            or os.getenv(self.ENV_FALLBACK_BASE_URL)
            or self.DEFAULT_BASE_URL
        )
        return raw.strip().rstrip("/")

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model = OpenAIEmbeddings(
                model=self._model_name,
                openai_api_base=self._resolve_base_url(),
                openai_api_key=os.getenv(self.ENV_API_KEY) or "EMPTY",
                check_embedding_ctx_length=False,
            )

    def _resolve_dim(self) -> None:
        if self._dim is None:
            self._ensure_model()
            test_vec = self._model.embed_query("probe-dimension")
            self._dim = int(len(test_vec)) if test_vec else 0

    def dimension(self) -> int:
        self._resolve_dim()
        assert self._dim is not None
        return self._dim

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            self._resolve_dim()
            assert self._dim is not None
            return np.empty((0, self._dim), dtype=np.float32)

        self._ensure_model()
        vecs = self._model.embed_documents(list(texts))
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)
        return arr

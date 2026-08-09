# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.providers._openai_compatible import (
    embed_openai_compatible,
)


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
        self._client: Optional[OpenAI] = None
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

    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=self._resolve_base_url(),
                api_key=os.getenv(self.ENV_API_KEY) or "EMPTY",
            )
        return self._client

    def _resolve_dim(self) -> None:
        if self._dim is None:
            vectors = embed_openai_compatible(
                self._ensure_client(),
                model=self._model_name,
                texts=["probe-dimension"],
            )
            self._dim = int(vectors.shape[1])

    def dimension(self) -> int:
        self._resolve_dim()
        assert self._dim is not None
        return self._dim

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        batch = list(texts)
        if not batch:
            self._resolve_dim()
            assert self._dim is not None
            return np.empty((0, self._dim), dtype=np.float32)

        vectors = embed_openai_compatible(
            self._ensure_client(),
            model=self._model_name,
            texts=batch,
        )
        if self._dim is None:
            self._dim = int(vectors.shape[1])
        elif vectors.shape[1] != self._dim:
            raise ValueError("Embedding dimension changed for provider instance")
        return vectors

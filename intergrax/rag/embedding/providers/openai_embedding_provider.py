# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.providers._openai_compatible import (
    embed_openai_compatible,
)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider.
    """

    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or self.DEFAULT_MODEL
        self._client: Optional[OpenAI] = None
        self._dim: Optional[int] = None

    def provider_name(self) -> str:
        return "openai"

    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
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
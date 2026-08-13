# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider

if TYPE_CHECKING:
    from langchain_ollama import OllamaEmbeddings


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Ollama embedding provider.
    """

    DEFAULT_MODEL = "nomic-embed-text"

    def __init__(
        self,
        model_name: Optional[str] = None
    ) -> None:

        self._model_name = model_name or self.DEFAULT_MODEL
        self._model: Optional[OllamaEmbeddings] = None

        self._dim: Optional[int] = None

    def provider_name(self) -> str:
        return "ollama"

    def _ensure_model(self) -> None:

        if self._model is None:
            try:
                from langchain_ollama import OllamaEmbeddings
            except ModuleNotFoundError as exc:
                if exc.name == "langchain_ollama":
                    raise RuntimeError(
                        "Provider 'ollama' requires optional dependency group "
                        "'rag-langchain-embeddings'. Install Intergrax with "
                        "'rag-langchain-embeddings'."
                    ) from exc
                raise

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
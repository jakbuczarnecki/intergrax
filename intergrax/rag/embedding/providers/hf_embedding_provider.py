# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
from numpy.typing import NDArray

from sentence_transformers import SentenceTransformer

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider


class HFEmbeddingProvider(EmbeddingProvider):
    """
    HuggingFace SentenceTransformer embedding provider.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize_inside: bool = False,
        max_length: Optional[int] = None,
    ) -> None:

        self._model_name = model_name or self.DEFAULT_MODEL
        self._device = device
        self._max_length = max_length

        self._model: Optional[SentenceTransformer] = None

        self._batch_size = int(batch_size)
        self._normalize_inside = bool(normalize_inside)

        self._dim: Optional[int] = None

    def provider_name(self) -> str:
        return "hf"

    def configured_device(self) -> str | None:
        return self._device

    def resolved_device(self) -> str | None:
        if self._model is None:
            return self._device
        return str(self._model.device)

    def _ensure_model(self) -> None:

        if self._model is None:

            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
            )

            if self._max_length is not None:
                self._model.max_seq_length = int(self._max_length)

    def _resolve_dim(self) -> None:

        if self._dim is None:

            self._ensure_model()

            self._dim = int(self._model.get_sentence_embedding_dimension())

    def dimension(self) -> int:

        self._resolve_dim()
        return self._dim

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:

        if not texts:
            self._resolve_dim()
            return np.empty((0, self._dim), dtype=np.float32)

        self._ensure_model()

        vecs = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_inside,
        )

        return vecs.astype(np.float32)
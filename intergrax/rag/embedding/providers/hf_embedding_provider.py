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

    def __init__(
        self,
        model_name: str,
        *,
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize_inside: bool = False,
        max_length: Optional[int] = None,
    ) -> None:

        self._model = SentenceTransformer(model_name, device=device)

        if max_length is not None:
            try:
                self._model.max_seq_length = int(max_length)
            except Exception:
                pass

        self._batch_size = int(batch_size)
        self._normalize_inside = bool(normalize_inside)

        self._dim = int(self._model.get_sentence_embedding_dimension())

    def provider_name(self) -> str:
        return "hf"

    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:

        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        vecs = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_inside,
        )

        return vecs.astype(np.float32)
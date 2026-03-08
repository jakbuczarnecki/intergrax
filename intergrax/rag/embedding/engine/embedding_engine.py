# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence, Iterable, List

import numpy as np
from numpy.typing import NDArray

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.registry.embedding_provider_registry import EmbeddingProviderRegistry

class EmbeddingEngine:
    """
    Execution engine responsible for generating embeddings
    using a provider resolved from the EmbeddingProviderRegistry.

    Responsibilities
    ----------------
    - provider resolution
    - batching
    - retry
    - optional normalization
    """

    def __init__(
        self,
        registry: EmbeddingProviderRegistry,
        *,
        batch_size: int = 64,
        normalize: bool = False,
        max_retries: int = 2,
    ) -> None:
        self._registry = registry
        self._batch_size = int(batch_size)
        self._normalize = bool(normalize)
        self._max_retries = int(max_retries)

    def embed(
        self,
        texts: Sequence[str],
        provider_id: str,
    ) -> NDArray[np.float32]:
        """
        Generate embeddings for a batch of texts.

        Parameters
        ----------
        texts : Sequence[str]
            Text inputs.

        provider_id : str
            Provider identifier registered in the registry.

        Returns
        -------
        NDArray[np.float32]
            Embedding matrix with shape (N, dim)
        """

        provider: EmbeddingProvider = self._registry.get(provider_id)

        if not texts:
            return np.empty((0, provider.dimension()), dtype=np.float32)

        results: List[NDArray[np.float32]] = []

        for batch in self._batch_iter(texts):
            vecs = self._embed_with_retry(provider, batch)
            results.append(vecs)

        arr = np.vstack(results)

        if self._normalize:
            arr = self._normalize_vectors(arr)

        return arr

    def _batch_iter(
        self,
        texts: Sequence[str],
    ) -> Iterable[Sequence[str]]:

        for i in range(0, len(texts), self._batch_size):
            yield texts[i : i + self._batch_size]

    def _embed_with_retry(
        self,
        provider: EmbeddingProvider,
        texts: Sequence[str],
    ) -> NDArray[np.float32]:

        last_exc: Exception | None = None

        for _ in range(self._max_retries + 1):
            try:
                return provider.embed(texts)
            except Exception as exc:
                last_exc = exc

        raise RuntimeError("Embedding failed after retries") from last_exc

    @staticmethod
    def _normalize_vectors(
        arr: NDArray[np.float32],
    ) -> NDArray[np.float32]:

        norms = np.linalg.norm(arr, axis=1, keepdims=True)

        norms[norms == 0] = 1.0

        return arr / norms
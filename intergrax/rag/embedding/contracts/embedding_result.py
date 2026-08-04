# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


from intergrax.knowledge.contracts import KnowledgeDocument


@dataclass(frozen=True)
class EmbeddingResult:
    """Immutable, validated alignment of native documents and vectors."""

    documents: tuple[KnowledgeDocument, ...]
    embeddings: NDArray[np.float32]

    def __post_init__(self) -> None:
        documents = tuple(self.documents)
        for document in documents:
            if not isinstance(document, KnowledgeDocument):
                raise TypeError("documents must contain only KnowledgeDocument values")
        object.__setattr__(self, "documents", documents)

        if not isinstance(self.embeddings, np.ndarray):
            raise TypeError("embeddings must be a numpy.ndarray")

        try:
            embeddings = np.array(self.embeddings, dtype=np.float32, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("embeddings must be numeric") from exc

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional matrix")
        if len(documents) == 0:
            if embeddings.shape != (0, 0):
                raise ValueError("empty results must use embeddings with shape (0, 0)")
        elif embeddings.shape[0] != len(documents):
            raise ValueError("embedding rows must match the document count")
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings must contain only finite values")

        embeddings.setflags(write=False)
        object.__setattr__(self, "embeddings", embeddings)
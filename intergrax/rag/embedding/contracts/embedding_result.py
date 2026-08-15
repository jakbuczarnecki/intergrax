# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


from intergrax.knowledge.contracts import KnowledgeDocument


def validate_embedding_matrix(
    embeddings: object,
    *,
    expected_rows: int,
) -> NDArray[np.float32]:
    """Validate and normalize an embedding matrix for vector-store use."""
    if type(expected_rows) is not int:
        raise TypeError("expected_rows must be an exact int, not bool")
    if expected_rows < 0:
        raise ValueError("expected_rows must be non-negative")
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("embeddings must be a numpy.ndarray")

    try:
        normalized = np.array(embeddings, dtype=np.float32, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("embeddings must be numeric") from exc

    if normalized.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    if normalized.shape[0] != expected_rows:
        raise ValueError("embedding rows must match expected_rows")
    if expected_rows == 0 and normalized.shape != (0, 0):
        raise ValueError("empty results must use embeddings with shape (0, 0)")
    if expected_rows > 0 and normalized.shape[1] <= 0:
        raise ValueError("non-empty results must have a positive embedding dimension")
    if not np.isfinite(normalized).all():
        raise ValueError("embeddings must contain only finite values")

    normalized.setflags(write=False)
    return normalized


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

        embeddings = validate_embedding_matrix(
            self.embeddings,
            expected_rows=len(documents),
        )
        object.__setattr__(self, "embeddings", embeddings)
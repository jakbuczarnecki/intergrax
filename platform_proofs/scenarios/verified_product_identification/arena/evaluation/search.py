"""In-memory cosine retrieval for arena quality evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def validate_embedding_matrix(matrix: NDArray[np.float64]) -> None:
    if matrix.ndim != 2:
        msg = f"embedding matrix must be 2D, got ndim={matrix.ndim}"
        raise ValueError(msg)
    if matrix.shape[0] == 0:
        msg = "embedding matrix must have at least one row"
        raise ValueError(msg)
    if not np.isfinite(matrix).all():
        msg = "embedding matrix contains non-finite values"
        raise ValueError(msg)


def validate_query_vector(vector: NDArray[np.float64], *, expected_dimension: int) -> None:
    if vector.ndim != 1:
        msg = f"query vector must be 1D, got ndim={vector.ndim}"
        raise ValueError(msg)
    if vector.shape[0] != expected_dimension:
        msg = (
            f"query dimension {vector.shape[0]} does not match corpus dimension "
            f"{expected_dimension}"
        )
        raise ValueError(msg)
    if not np.isfinite(vector).all():
        msg = "query vector contains non-finite values"
        raise ValueError(msg)


def rank_corpus_by_cosine_similarity(
    corpus_embeddings: NDArray[np.float64],
    query_embedding: NDArray[np.float64],
    *,
    top_k: int,
) -> tuple[int, ...]:
    validate_embedding_matrix(corpus_embeddings)
    validate_query_vector(query_embedding, expected_dimension=corpus_embeddings.shape[1])
    if top_k <= 0:
        msg = "top_k must be > 0"
        raise ValueError(msg)

    corpus_norms = np.linalg.norm(corpus_embeddings, axis=1)
    query_norm = float(np.linalg.norm(query_embedding))
    if query_norm == 0.0 or np.any(corpus_norms == 0.0):
        msg = "zero-norm embeddings are not supported for cosine ranking"
        raise ValueError(msg)

    scores = corpus_embeddings @ query_embedding / (corpus_norms * query_norm)
    limit = min(top_k, scores.shape[0])
    top_indices = np.argpartition(-scores, limit - 1)[:limit]
    ordered = top_indices[np.argsort(-scores[top_indices])]
    return tuple(int(index) for index in ordered)

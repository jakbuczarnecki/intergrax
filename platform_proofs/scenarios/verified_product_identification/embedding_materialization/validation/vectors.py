"""Embedding batch vector validation — provider-neutral."""

from __future__ import annotations

import math

from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.errors import (
    EmbeddingMaterializationProviderError,
)


def validate_embedding_batch_vectors(
    *,
    vectors: tuple[tuple[float, ...], ...],
    expected_count: int,
    expected_dimension: int,
) -> None:
    if len(vectors) != expected_count:
        raise EmbeddingMaterializationProviderError(
            f"expected {expected_count} vectors, got {len(vectors)}"
        )
    for row_index, vector in enumerate(vectors):
        if len(vector) != expected_dimension:
            raise EmbeddingMaterializationProviderError(
                f"vector at index {row_index} has dimension {len(vector)}, "
                f"expected {expected_dimension}"
            )
        for value_index, value in enumerate(vector):
            if not math.isfinite(value):
                raise EmbeddingMaterializationProviderError(
                    f"vector at index {row_index} contains non-finite value "
                    f"at position {value_index}"
                )
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0 or math.isclose(norm, 0.0):
            raise EmbeddingMaterializationProviderError(
                f"vector at index {row_index} is empty or zero-norm"
            )

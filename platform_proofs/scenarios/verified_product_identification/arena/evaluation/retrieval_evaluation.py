"""Stage-scoped retrieval quality evaluation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from platform_proofs.scenarios.verified_product_identification.arena.contracts.errors import (
    EmbeddingArenaEvaluationScopeError,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.results import (
    RetrievalQualityMetrics,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.stage_evaluation_scope import (
    EmbeddingArenaStageEvaluationScope,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.ground_truth import (
    resolve_relevant_indices_or_fail,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.metrics import (
    aggregate_retrieval_metrics,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.search import (
    rank_corpus_by_cosine_similarity,
)


def _validate_embedding_matrix(
    matrix: NDArray[np.float64],
    *,
    row_count: int,
    expected_dimension: int,
    label: str,
) -> None:
    if matrix.ndim != 2:
        msg = f"{label} must be 2D, got ndim={matrix.ndim}"
        raise EmbeddingArenaEvaluationScopeError(msg)
    if matrix.shape[0] != row_count:
        msg = f"{label} row count {matrix.shape[0]} != expected {row_count}"
        raise EmbeddingArenaEvaluationScopeError(msg)
    if matrix.shape[1] != expected_dimension:
        msg = (
            f"{label} dimension {matrix.shape[1]} != expected {expected_dimension}"
        )
        raise EmbeddingArenaEvaluationScopeError(msg)
    if not np.isfinite(matrix).all():
        msg = f"{label} contains non-finite values"
        raise EmbeddingArenaEvaluationScopeError(msg)


def evaluate_retrieval_quality_for_scope(
    *,
    scope: EmbeddingArenaStageEvaluationScope,
    corpus_embeddings: NDArray[np.float64],
    query_embeddings: NDArray[np.float64],
    expected_dimension: int,
) -> RetrievalQualityMetrics:
    if corpus_embeddings.size == 0:
        msg = "corpus_embeddings must not be empty"
        raise EmbeddingArenaEvaluationScopeError(msg)

    _validate_embedding_matrix(
        corpus_embeddings,
        row_count=scope.corpus_size,
        expected_dimension=expected_dimension,
        label="corpus_embeddings",
    )
    _validate_embedding_matrix(
        query_embeddings,
        row_count=len(scope.query_cases),
        expected_dimension=expected_dimension,
        label="query_embeddings",
    )

    per_query_relevant: list[list[int]] = []
    per_query_ranked: list[list[int]] = []
    top_k = min(10, corpus_embeddings.shape[0])

    for case_index, case in enumerate(scope.query_cases):
        relevant = list(resolve_relevant_indices_or_fail(case, scope.offer_index))
        query_vector = query_embeddings[case_index]
        ranked = rank_corpus_by_cosine_similarity(
            corpus_embeddings,
            query_vector,
            top_k=top_k,
        )
        per_query_relevant.append(relevant)
        per_query_ranked.append(list(ranked))

    return aggregate_retrieval_metrics(per_query_relevant, per_query_ranked)

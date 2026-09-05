"""Deterministic retrieval ranking metrics for arena evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence

from platform_proofs.scenarios.verified_product_identification.arena.contracts.results import (
    RetrievalQualityMetrics,
)


def _relevant_set(relevant_indices: Sequence[int]) -> frozenset[int]:
    return frozenset(relevant_indices)


def recall_at_k(
    relevant_indices: Sequence[int],
    ranked_indices: Sequence[int],
    k: int,
) -> float:
    if k <= 0:
        msg = "k must be > 0"
        raise ValueError(msg)
    relevant = _relevant_set(relevant_indices)
    if not relevant:
        return 0.0
    top_k = ranked_indices[:k]
    hits = sum(1 for index in top_k if index in relevant)
    return hits / float(len(relevant))


def mrr_at_k(
    relevant_indices: Sequence[int],
    ranked_indices: Sequence[int],
    k: int,
) -> float:
    if k <= 0:
        msg = "k must be > 0"
        raise ValueError(msg)
    relevant = _relevant_set(relevant_indices)
    if not relevant:
        return 0.0
    for rank, index in enumerate(ranked_indices[:k], start=1):
        if index in relevant:
            return 1.0 / float(rank)
    return 0.0


def ndcg_at_k(
    relevant_indices: Sequence[int],
    ranked_indices: Sequence[int],
    k: int,
) -> float:
    if k <= 0:
        msg = "k must be > 0"
        raise ValueError(msg)
    relevant = _relevant_set(relevant_indices)
    if not relevant:
        return 0.0

    def dcg(ranked: Sequence[int]) -> float:
        score = 0.0
        for rank, index in enumerate(ranked[:k], start=1):
            if index in relevant:
                score += 1.0 / math.log2(rank + 1.0)
        return score

    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))
    if ideal_dcg == 0.0:
        return 0.0
    return dcg(ranked_indices) / ideal_dcg


def aggregate_retrieval_metrics(
    per_query_relevant: Sequence[Sequence[int]],
    per_query_ranked: Sequence[Sequence[int]],
) -> RetrievalQualityMetrics:
    if len(per_query_relevant) != len(per_query_ranked):
        msg = "per_query_relevant and per_query_ranked must have equal length"
        raise ValueError(msg)
    if not per_query_relevant:
        msg = "at least one query is required"
        raise ValueError(msg)

    recall_1_values: list[float] = []
    recall_5_values: list[float] = []
    recall_10_values: list[float] = []
    mrr_values: list[float] = []
    ndcg_values: list[float] = []

    for relevant, ranked in zip(per_query_relevant, per_query_ranked, strict=True):
        recall_1_values.append(recall_at_k(relevant, ranked, 1))
        recall_5_values.append(recall_at_k(relevant, ranked, 5))
        recall_10_values.append(recall_at_k(relevant, ranked, 10))
        mrr_values.append(mrr_at_k(relevant, ranked, 10))
        ndcg_values.append(ndcg_at_k(relevant, ranked, 10))

    count = len(per_query_relevant)
    return RetrievalQualityMetrics(
        recall_at_1=sum(recall_1_values) / count,
        recall_at_5=sum(recall_5_values) / count,
        recall_at_10=sum(recall_10_values) / count,
        mrr_at_10=sum(mrr_values) / count,
        ndcg_at_10=sum(ndcg_values) / count,
        query_count=count,
    )

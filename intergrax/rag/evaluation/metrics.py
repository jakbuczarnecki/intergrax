# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Offline RAG retrieval metrics for CI regression gates."""

from __future__ import annotations

from typing import Iterable, Sequence, Set


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    if not relevant_ids or k <= 0:
        return 0.0
    top = retrieved_ids[:k]
    hits = sum(1 for doc_id in top if doc_id in relevant_ids)
    return hits / float(len(relevant_ids))


def mean_reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
) -> float:
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in rel:
            return 1.0 / float(rank)
    return 0.0


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    if k <= 0:
        return 0.0
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for doc_id in top if doc_id in relevant_ids)
    return hits / float(len(top))


def _dcg_at_k(relevance_scores: Sequence[float], k: int) -> float:
    total = 0.0
    for index, rel in enumerate(relevance_scores[:k], start=1):
        if rel <= 0.0:
            continue
        total += rel / __log2(1.0 + index)
    return total


def __log2(value: float) -> float:
    import math

    return math.log2(value)


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """
    Normalized discounted cumulative gain at ``k``.

    Relevant documents receive a binary gain of 1.0; ideal DCG assumes all
    relevant docs appear first.
    """
    if k <= 0 or not relevant_ids:
        return 0.0
    gains = [1.0 if doc_id in relevant_ids else 0.0 for doc_id in retrieved_ids[:k]]
    dcg = _dcg_at_k(gains, k)
    ideal_gains = [1.0] * min(len(relevant_ids), k)
    idcg = _dcg_at_k(ideal_gains, k)
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg

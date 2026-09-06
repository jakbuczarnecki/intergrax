"""Single-relevant-item retrieval metrics for proof evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SingleRelevantRank:
    found: bool
    zero_based_rank: int | None


@dataclass(frozen=True, slots=True)
class SingleRelevantRetrievalMetrics:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    relevant_rank: int | None


def locate_single_relevant_rank(
    ranked_candidate_ids: Sequence[str],
    expected_candidate_id: str,
) -> SingleRelevantRank:
    for index, candidate_id in enumerate(ranked_candidate_ids):
        if candidate_id == expected_candidate_id:
            return SingleRelevantRank(found=True, zero_based_rank=index)
    return SingleRelevantRank(found=False, zero_based_rank=None)


def single_relevant_recall_at_k(rank: SingleRelevantRank, k: int) -> float:
    if k <= 0:
        msg = "k must be > 0"
        raise ValueError(msg)
    if not rank.found or rank.zero_based_rank is None:
        return 0.0
    return 1.0 if rank.zero_based_rank < k else 0.0


def single_relevant_mrr_at_k(rank: SingleRelevantRank, k: int) -> float:
    if k <= 0:
        msg = "k must be > 0"
        raise ValueError(msg)
    if not rank.found or rank.zero_based_rank is None:
        return 0.0
    one_based_rank = rank.zero_based_rank + 1
    if one_based_rank > k:
        return 0.0
    return 1.0 / float(one_based_rank)


def single_relevant_ndcg_at_k(rank: SingleRelevantRank, k: int) -> float:
    if k <= 0:
        msg = "k must be > 0"
        raise ValueError(msg)
    if not rank.found or rank.zero_based_rank is None:
        return 0.0
    one_based_rank = rank.zero_based_rank + 1
    if one_based_rank > k:
        return 0.0
    dcg = 1.0 / math.log2(one_based_rank + 1.0)
    return dcg


def evaluate_single_relevant_ranking(
    ranked_candidate_ids: Sequence[str],
    expected_candidate_id: str,
) -> SingleRelevantRetrievalMetrics:
    rank = locate_single_relevant_rank(ranked_candidate_ids, expected_candidate_id)
    one_based_rank = (
        rank.zero_based_rank + 1
        if rank.found and rank.zero_based_rank is not None
        else None
    )
    return SingleRelevantRetrievalMetrics(
        recall_at_1=single_relevant_recall_at_k(rank, 1),
        recall_at_5=single_relevant_recall_at_k(rank, 5),
        recall_at_10=single_relevant_recall_at_k(rank, 10),
        mrr_at_10=single_relevant_mrr_at_k(rank, 10),
        ndcg_at_10=single_relevant_ndcg_at_k(rank, 10),
        relevant_rank=one_based_rank,
    )

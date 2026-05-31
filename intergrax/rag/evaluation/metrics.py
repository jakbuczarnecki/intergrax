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

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Retriever degrade chain after retry exhaustion (M-RAG.28)."""

from __future__ import annotations

from typing import Iterable, List, Sequence

CANONICAL_RETRIEVER_FALLBACK_CHAIN: tuple[str, ...] = (
    "fusion",
    "hybrid",
    "vector_similarity",
)


def retriever_fallback_chain(
    primary_retriever_id: str,
    registered_retriever_ids: Iterable[str],
) -> List[str]:
    """
    Build ordered retriever ids: primary first, then canonical degrade tail.

    Example: ``fusion`` → ``hybrid`` → ``vector_similarity`` when all are registered.
    """
    registered = set(registered_retriever_ids)
    ordered: List[str] = []

    def _append(retriever_id: str) -> None:
        if retriever_id in registered and retriever_id not in ordered:
            ordered.append(retriever_id)

    _append(primary_retriever_id)
    if primary_retriever_id in CANONICAL_RETRIEVER_FALLBACK_CHAIN:
        start = CANONICAL_RETRIEVER_FALLBACK_CHAIN.index(primary_retriever_id) + 1
        for retriever_id in CANONICAL_RETRIEVER_FALLBACK_CHAIN[start:]:
            _append(retriever_id)
    else:
        for retriever_id in CANONICAL_RETRIEVER_FALLBACK_CHAIN:
            _append(retriever_id)

    if not ordered and primary_retriever_id:
        return [primary_retriever_id]
    return ordered


def merge_attempted_retriever_ids(*groups: Sequence[str]) -> tuple[str, ...]:
    merged: list[str] = []
    for group in groups:
        for retriever_id in group:
            if retriever_id and retriever_id not in merged:
                merged.append(retriever_id)
    return tuple(merged)

# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Canonical reciprocal rank fusion math (0-based channel ranks)."""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_contribution(*, rank: int, rrf_k: int) -> float:
    """One channel term: ``1 / (rrf_k + rank + 1)`` with zero-based ``rank``."""

    if type(rank) is not int or rank < 0:
        raise ValueError("rank must be a non-negative int")
    if type(rrf_k) is not int or rrf_k <= 0:
        raise ValueError("rrf_k must be a positive int")
    return 1.0 / (rrf_k + rank + 1)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Fuse ranked document-id lists using RRF.

    Duplicate ids within one channel keep only the best (minimum) rank.
    Tie-break: fused score DESC, best channel rank ASC, candidate id ASC.
    """

    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive int")

    scores: dict[str, float] = {}
    best_channel_rank: dict[str, int] = {}

    for ranked in ranked_lists:
        best_in_channel: dict[str, int] = {}
        for rank, candidate_id in enumerate(ranked):
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError("candidate_id must be a non-empty str")
            previous = best_in_channel.get(candidate_id)
            if previous is not None and rank >= previous:
                continue
            best_in_channel[candidate_id] = rank

        for candidate_id, rank in best_in_channel.items():
            contribution = reciprocal_rank_contribution(rank=rank, rrf_k=k)
            scores[candidate_id] = scores.get(candidate_id, 0.0) + contribution
            current_best = best_channel_rank.get(candidate_id)
            if current_best is None or rank < current_best:
                best_channel_rank[candidate_id] = rank

    ordered = sorted(
        scores.items(),
        key=lambda item: (-item[1], best_channel_rank[item[0]], item[0]),
    )
    return ordered

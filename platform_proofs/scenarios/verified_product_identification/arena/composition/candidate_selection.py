"""Resolve arena candidates from registry with optional typed subset."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.arena.composition.candidates import (
    build_default_arena_candidates,
)
from platform_proofs.scenarios.verified_product_identification.arena.composition.finalist_qualification_policy import (
    FINALIST_BGE_QWEN_CANDIDATE_SELECTION,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.candidate_selection import (
    EmbeddingArenaCandidateSelection,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.candidates import (
    EmbeddingArenaCandidate,
)

__all__ = (
    "FINALIST_BGE_QWEN_CANDIDATE_SELECTION",
    "resolve_arena_candidates",
)


def resolve_arena_candidates(
    *,
    include_e5_control: bool = False,
    selection: EmbeddingArenaCandidateSelection | None = None,
) -> tuple[EmbeddingArenaCandidate, ...]:
    all_candidates = build_default_arena_candidates(include_e5_control=include_e5_control)
    if selection is None:
        return all_candidates

    by_id = {candidate.candidate_id: candidate for candidate in all_candidates}
    resolved: list[EmbeddingArenaCandidate] = []
    for candidate_id in selection.candidate_ids:
        try:
            resolved.append(by_id[candidate_id])
        except KeyError as exc:
            msg = f"unknown arena candidate id in selection: {candidate_id!r}"
            raise ValueError(msg) from exc
    return tuple(resolved)

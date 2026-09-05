"""Benchmark ground-truth resolution — explicit validation, fail closed."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from platform_proofs.scenarios.verified_product_identification.arena.contracts.errors import (
    EmbeddingArenaBenchmarkGroundTruthError,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.query_benchmark import (
    ArenaSourceRef,
    EmbeddingArenaQueryCase,
)


def resolve_relevant_indices_or_fail(
    case: EmbeddingArenaQueryCase,
    offer_index: Mapping[str, int],
) -> tuple[int, ...]:
    if not case.relevant_source_refs:
        msg = f"case {case.case_id} has no relevant_source_refs"
        raise EmbeddingArenaBenchmarkGroundTruthError(msg)

    seen_offer_ids: set[str] = set()
    indices: list[int] = []
    for source_ref in case.relevant_source_refs:
        if source_ref.offer_id in seen_offer_ids:
            msg = (
                f"case {case.case_id} has duplicate relevant offer_id "
                f"{source_ref.offer_id}"
            )
            raise EmbeddingArenaBenchmarkGroundTruthError(msg)
        seen_offer_ids.add(source_ref.offer_id)
        if source_ref.offer_id not in offer_index:
            msg = (
                f"case {case.case_id} references offer_id {source_ref.offer_id} "
                "outside the stage corpus"
            )
            raise EmbeddingArenaBenchmarkGroundTruthError(msg)
        indices.append(offer_index[source_ref.offer_id])

    return tuple(indices)


def validate_query_cases_against_offer_index(
    query_cases: Sequence[EmbeddingArenaQueryCase],
    offer_index: Mapping[str, int],
) -> None:
    for case in query_cases:
        resolve_relevant_indices_or_fail(case, offer_index)


def validate_source_refs_in_corpus(
    source_refs: Sequence[ArenaSourceRef],
    offer_index: Mapping[str, int],
    *,
    case_id: str,
) -> None:
    for source_ref in source_refs:
        if source_ref.offer_id not in offer_index:
            msg = (
                f"case {case_id} references offer_id {source_ref.offer_id} "
                "outside the stage corpus"
            )
            raise EmbeddingArenaBenchmarkGroundTruthError(msg)

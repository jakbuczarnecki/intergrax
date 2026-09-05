"""Build validated stage-local evaluation scopes."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.arena.contracts.errors import (
    EmbeddingArenaEvaluationScopeError,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.stage_evaluation_scope import (
    EmbeddingArenaStageEvaluationScope,
    compute_stage_content_fingerprint,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.versioning import (
    ARENA_QUERY_BENCHMARK_VERSION,
    ARENA_SAMPLE_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.ground_truth import (
    validate_query_cases_against_offer_index,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.query_builder import (
    build_query_benchmark_cases,
)
from platform_proofs.scenarios.verified_product_identification.arena.sampling.arena_sample import (
    ArenaSampleRecord,
)


def build_stage_evaluation_scope(
    *,
    stage_name: str,
    records: tuple[ArenaSampleRecord, ...],
) -> EmbeddingArenaStageEvaluationScope:
    if not records:
        msg = "records must not be empty"
        raise EmbeddingArenaEvaluationScopeError(msg)

    offer_index = {record.offer_id: index for index, record in enumerate(records)}
    query_cases = build_query_benchmark_cases(records)
    if not query_cases:
        msg = f"no query benchmark cases for stage {stage_name}"
        raise EmbeddingArenaEvaluationScopeError(msg)

    validate_query_cases_against_offer_index(query_cases, offer_index)

    return EmbeddingArenaStageEvaluationScope(
        stage_name=stage_name,
        records=records,
        query_cases=query_cases,
        offer_index=offer_index,
        corpus_size=len(records),
        benchmark_version=ARENA_QUERY_BENCHMARK_VERSION,
        sample_version=ARENA_SAMPLE_VERSION,
        content_fingerprint=compute_stage_content_fingerprint(records),
    )

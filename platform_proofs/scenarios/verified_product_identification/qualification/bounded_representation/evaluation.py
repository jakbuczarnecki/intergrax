"""Retrieval quality evaluation for bounded product representation variants."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from platform_proofs.scenarios.verified_product_identification.arena.contracts.query_benchmark import (
    EmbeddingArenaQueryCase,
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
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.search import (
    rank_corpus_by_cosine_similarity,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_representation.contracts import (
    PerQueryRetrievalComparison,
    RepresentationVariant,
    VariantQualityGateResult,
)

_MRR_DEGRADATION_TOLERANCE = 0.01
_NDCG_DEGRADATION_TOLERANCE = 0.01


def _relevant_rank(
    ranked_indices: Sequence[int],
    relevant_indices: Sequence[int],
) -> int | None:
    relevant = frozenset(relevant_indices)
    for rank, index in enumerate(ranked_indices, start=1):
        if index in relevant:
            return rank
    return None


def evaluate_variant_rankings(
    scope: EmbeddingArenaStageEvaluationScope,
    *,
    corpus_embeddings: NDArray[np.float64],
    query_embeddings: NDArray[np.float64],
) -> tuple[RetrievalQualityMetrics, tuple[tuple[int, ...], ...]]:
    top_k = min(10, corpus_embeddings.shape[0])
    per_query_relevant: list[list[int]] = []
    per_query_ranked: list[list[int]] = []

    for case_index, case in enumerate(scope.query_cases):
        relevant = list(resolve_relevant_indices_or_fail(case, scope.offer_index))
        ranked = rank_corpus_by_cosine_similarity(
            corpus_embeddings,
            query_embeddings[case_index],
            top_k=top_k,
        )
        per_query_relevant.append(relevant)
        per_query_ranked.append(list(ranked))

    metrics = aggregate_retrieval_metrics(per_query_relevant, per_query_ranked)
    return metrics, tuple(tuple(ranked) for ranked in per_query_ranked)


def compare_query_rankings(
    scope: EmbeddingArenaStageEvaluationScope,
    *,
    full_ranked: Sequence[Sequence[int]],
    candidate_ranked: Sequence[Sequence[int]],
) -> tuple[PerQueryRetrievalComparison, ...]:
    if len(full_ranked) != len(candidate_ranked):
        msg = "full_ranked and candidate_ranked must have equal length"
        raise ValueError(msg)
    if len(full_ranked) != len(scope.query_cases):
        msg = "ranking count must match query case count"
        raise ValueError(msg)

    comparisons: list[PerQueryRetrievalComparison] = []
    for case, full_case_ranked, candidate_case_ranked in zip(
        scope.query_cases,
        full_ranked,
        candidate_ranked,
        strict=True,
    ):
        relevant = resolve_relevant_indices_or_fail(case, scope.offer_index)
        expected_offer_id = case.relevant_source_refs[0].offer_id
        full_rank = _relevant_rank(full_case_ranked, relevant)
        candidate_rank = _relevant_rank(candidate_case_ranked, relevant)
        rank_delta = None
        if full_rank is not None and candidate_rank is not None:
            rank_delta = candidate_rank - full_rank
        top1_changed = False
        if full_case_ranked and candidate_case_ranked:
            top1_changed = full_case_ranked[0] != candidate_case_ranked[0]
        expected_lost_from_top5 = (
            full_rank is not None
            and full_rank <= 5
            and (candidate_rank is None or candidate_rank > 5)
        )
        expected_lost_from_top10 = (
            full_rank is not None
            and full_rank <= 10
            and (candidate_rank is None or candidate_rank > 10)
        )
        is_severe_regression = (
            full_rank is not None
            and full_rank <= 5
            and (candidate_rank is None or candidate_rank > 10)
        )
        is_top1_regression = full_rank == 1 and (candidate_rank is None or candidate_rank > 1)
        comparisons.append(
            PerQueryRetrievalComparison(
                query_id=case.case_id,
                expected_offer_id=expected_offer_id,
                full_rank=full_rank,
                candidate_rank=candidate_rank,
                rank_delta=rank_delta,
                top1_changed=top1_changed,
                expected_lost_from_top5=expected_lost_from_top5,
                expected_lost_from_top10=expected_lost_from_top10,
                is_severe_regression=is_severe_regression,
                is_top1_regression=is_top1_regression,
            )
        )
    return tuple(comparisons)


def evaluate_quality_gate(
    control: RetrievalQualityMetrics,
    candidate: RetrievalQualityMetrics,
    *,
    variant: RepresentationVariant,
    comparisons: Sequence[PerQueryRetrievalComparison],
) -> VariantQualityGateResult:
    failure_reasons: list[str] = []
    if candidate.recall_at_1 < control.recall_at_1:
        failure_reasons.append(
            f"Recall@1 {candidate.recall_at_1:.6f} < control {control.recall_at_1:.6f}"
        )
    if candidate.recall_at_5 < control.recall_at_5:
        failure_reasons.append(
            f"Recall@5 {candidate.recall_at_5:.6f} < control {control.recall_at_5:.6f}"
        )
    mrr_delta = control.mrr_at_10 - candidate.mrr_at_10
    if mrr_delta > _MRR_DEGRADATION_TOLERANCE:
        failure_reasons.append(
            f"MRR@10 degradation {mrr_delta:.6f} > {_MRR_DEGRADATION_TOLERANCE}"
        )
    ndcg_delta = control.ndcg_at_10 - candidate.ndcg_at_10
    if ndcg_delta > _NDCG_DEGRADATION_TOLERANCE:
        failure_reasons.append(
            f"nDCG@10 degradation {ndcg_delta:.6f} > {_NDCG_DEGRADATION_TOLERANCE}"
        )
    severe_regressions = tuple(
        comparison for comparison in comparisons if comparison.is_severe_regression
    )
    if severe_regressions:
        failure_reasons.append(f"severe regressions: {len(severe_regressions)}")
    return VariantQualityGateResult(
        variant=variant,
        metrics=candidate,
        passed=not failure_reasons,
        failure_reasons=tuple(failure_reasons),
    )


def select_winning_candidate(
    gates: dict[RepresentationVariant, VariantQualityGateResult],
) -> RepresentationVariant | None:
    passing_bounded = [
        variant
        for variant in (
            RepresentationVariant.TOKEN_LIMIT_512,
            RepresentationVariant.TOKEN_LIMIT_768,
            RepresentationVariant.TOKEN_LIMIT_1024,
        )
        if gates[variant].passed
    ]
    if not passing_bounded:
        return None
    return passing_bounded[0]


def per_query_metrics(
    relevant_indices: Sequence[int],
    ranked_indices: Sequence[int],
) -> tuple[float, float, float, float, float]:
    return (
        recall_at_k(relevant_indices, ranked_indices, 1),
        recall_at_k(relevant_indices, ranked_indices, 5),
        recall_at_k(relevant_indices, ranked_indices, 10),
        mrr_at_k(relevant_indices, ranked_indices, 10),
        ndcg_at_k(relevant_indices, ranked_indices, 10),
    )


def expected_offer_id_for_case(case: EmbeddingArenaQueryCase) -> str:
    return case.relevant_source_refs[0].offer_id

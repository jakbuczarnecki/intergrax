"""Unit tests for VPI embedding arena."""

from __future__ import annotations

import pytest

from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    WdcIdentifierEntry,
    WdcKeyValuePair,
    WdcSourceOffer,
)
from platform_proofs.scenarios.verified_product_identification.arena.composition.candidates import (
    build_default_arena_candidates,
)
from platform_proofs.scenarios.verified_product_identification.arena.composition.input_policies import (
    bge_m3_input_policy,
    nomic_v2_input_policy,
    qwen3_embedding_input_policy,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.classification import (
    EmbeddingArenaDecision,
    EmbeddingArenaVerdict,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.input_policy import (
    EmbeddingInputRole,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.results import (
    CandidateArenaResult,
    RetrievalQualityMetrics,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.estimates import (
    estimate_artifact_size,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.metrics import (
    aggregate_retrieval_metrics,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.query_builder import (
    build_query_benchmark_cases,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.search import (
    rank_corpus_by_cosine_similarity,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.verdict import (
    classify_candidate_verdict,
    classify_speedup_band,
    compute_quality_delta,
    compute_speedup_estimate,
    decide_arena_outcome,
    quality_non_regression_gate,
)
from platform_proofs.scenarios.verified_product_identification.arena.integration.reporting import (
    arena_report_to_json,
)
from platform_proofs.scenarios.verified_product_identification.arena.sampling.arena_sample import (
    ArenaSampleRecord,
    derive_strata_tags,
    select_arena_sample_records,
)

pytestmark = pytest.mark.unit


def _sample_offer(
    *,
    offer_id: str,
    title: str | None = "Example Product Title",
    brand: str | None = "BrandX",
    description: str | None = "Short description",
    identifiers: tuple[WdcIdentifierEntry, ...] = (),
    cluster_id: int | None = None,
) -> WdcSourceOffer:
    return WdcSourceOffer(
        offer_id=offer_id,
        cluster_id=cluster_id,
        category="electronics",
        identifiers=identifiers,
        title=title,
        description=description,
        brand=brand,
        price="9.99",
        key_value_pairs=(
            WdcKeyValuePair(source_key="color", source_value="black", raw_value="black"),
        ),
        spec_table_content="Voltage: 12V",
    )


def _sample_record(offer_id: str, *, cluster_id: int | None = None) -> ArenaSampleRecord:
    offer = _sample_offer(
        offer_id=offer_id,
        identifiers=(WdcIdentifierEntry(source_key="gtin", source_value=f"ID-{offer_id}"),),
        cluster_id=cluster_id,
    )
    return ArenaSampleRecord(
        offer_id=offer_id,
        global_row_index=int(offer_id.split("-")[-1]),
        semantic_text=f"{offer.brand} {offer.title} {offer.description}",
        source_offer=offer,
        strata_tags=derive_strata_tags(offer),
    )


def test_recall_mrr_ndcg_metrics() -> None:
    relevant = (0, 2)
    ranked = (2, 0, 1, 3)

    assert recall_at_k(relevant, ranked, 1) == 0.5
    assert recall_at_k(relevant, ranked, 5) == 1.0
    assert mrr_at_k(relevant, ranked, 10) == 1.0
    assert ndcg_at_k(relevant, ranked, 10) == pytest.approx(1.0)


def test_aggregate_metrics_handles_no_hit() -> None:
    metrics = aggregate_retrieval_metrics(
        per_query_relevant=((1,), (2,)),
        per_query_ranked=((0, 1), (0, 3)),
    )
    assert metrics.query_count == 2
    assert metrics.recall_at_1 == 0.0
    assert metrics.recall_at_10 == 0.5


def test_cosine_ranking_prefers_matching_vector() -> None:
    import numpy as np

    corpus = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
        ],
        dtype=np.float64,
    )
    query = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ranked = rank_corpus_by_cosine_similarity(corpus, query, top_k=2)
    assert ranked[0] == 0


def test_input_policies_do_not_mutate_semantic_text_for_documents() -> None:
    canonical = "brand title description"
    assert bge_m3_input_policy().transform(EmbeddingInputRole.DOCUMENT, canonical) == canonical
    assert (
        nomic_v2_input_policy().transform(EmbeddingInputRole.DOCUMENT, canonical)
        == f"search_document: {canonical}"
    )
    transformed = qwen3_embedding_input_policy().transform(EmbeddingInputRole.QUERY, canonical)
    assert canonical in transformed
    assert "Query:" in transformed


def test_query_builder_is_deterministic_and_has_no_cluster_leakage() -> None:
    records = tuple(_sample_record(f"offer-{index}", cluster_id=index % 2) for index in range(6))
    first = build_query_benchmark_cases(records, max_cases=20)
    second = build_query_benchmark_cases(records, max_cases=20)
    assert first == second
    for case in first:
        assert "cluster_id" not in case.query_text.casefold()


def test_sample_selection_is_deterministic() -> None:
    records = tuple(_sample_record(f"offer-{index}") for index in range(30))
    first = select_arena_sample_records(records, target_size=10)
    second = select_arena_sample_records(records, target_size=10)
    assert [record.offer_id for record in first] == [record.offer_id for record in second]


def test_speedup_and_quality_delta_helpers() -> None:
    speedup = compute_speedup_estimate(
        candidate_records_per_second=20.8,
        baseline_records_per_second=10.4,
        baseline_embedding_hours=99.0,
    )
    assert speedup.speedup_vs_baseline == pytest.approx(2.0)
    assert classify_speedup_band(speedup.speedup_vs_baseline) is not None

    baseline = RetrievalQualityMetrics(0.8, 0.9, 0.95, 0.7, 0.75, 10)
    candidate = RetrievalQualityMetrics(0.79, 0.89, 0.94, 0.69, 0.74, 10)
    delta = compute_quality_delta(candidate, baseline)
    assert quality_non_regression_gate(delta) is True


def test_verdict_and_decision_logic() -> None:
    verdict = classify_candidate_verdict(
        is_baseline=False,
        license_eligible=True,
        runtime_ok=True,
        correctness_ok=True,
        quality_delta=compute_quality_delta(
            RetrievalQualityMetrics(0.8, 0.9, 0.95, 0.7, 0.75, 10),
            RetrievalQualityMetrics(0.8, 0.9, 0.95, 0.7, 0.75, 10),
        ),
        speedup=compute_speedup_estimate(
            candidate_records_per_second=30.0,
            baseline_records_per_second=10.0,
            baseline_embedding_hours=100.0,
        ),
        long_input_regression=False,
    )
    assert verdict is EmbeddingArenaVerdict.WINNER_CANDIDATE

    baseline_result = CandidateArenaResult(
        candidate_id="bge-m3",
        verdict=EmbeddingArenaVerdict.BASELINE,
        runtime_metadata=None,
        truncation_profile=None,
        stage_a=None,
        stage_b=None,
        stage_c=None,
        quality_metrics=None,
        long_input_quality_metrics=None,
        quality_delta_vs_baseline=None,
        query_latency=None,
        artifact_size_estimate=None,
        full_build_estimate=None,
        speedup_estimate=None,
        warnings=(),
    )
    decision, _, finalists = decide_arena_outcome((baseline_result,))
    assert decision is EmbeddingArenaDecision.KEEP_BGE_M3
    assert finalists == ("bge-m3",)


def test_artifact_size_estimate() -> None:
    estimate = estimate_artifact_size(dimension=1024, record_count=3_770_377)
    assert estimate.bytes_per_vector == 4096
    assert estimate.preliminary_full_artifact_gb > 10.0


def test_default_candidates_include_required_set() -> None:
    candidates = build_default_arena_candidates(include_e5_control=True)
    ids = {candidate.candidate_id for candidate in candidates}
    assert {"bge-m3", "qwen3-0.6b", "nomic-v2-moe", "e5-large-instruct"} <= ids


def test_arena_report_json_roundtrip_minimal() -> None:
    from platform_proofs.scenarios.verified_product_identification.arena.contracts.results import (
        ArenaSampleManifest,
        EmbeddingArenaReport,
    )
    from platform_proofs.scenarios.verified_product_identification.arena.contracts.versioning import (
        ARENA_QUERY_BENCHMARK_VERSION,
        VPI_EMBEDDING_ARENA_VERSION,
    )

    report = EmbeddingArenaReport(
        arena_version=VPI_EMBEDDING_ARENA_VERSION,
        sample_manifest=ArenaSampleManifest(
            version="arena-sample-v1",
            selection_seed="seed",
            scan_row_limit=100,
            target_size=0,
            strata_quotas=(),
            records=(),
        ),
        query_benchmark_version=ARENA_QUERY_BENCHMARK_VERSION,
        query_cases=(),
        hardware=None,
        text_length_profile=None,
        candidate_results=(),
        decision=EmbeddingArenaDecision.MORE_EVIDENCE_REQUIRED,
        decision_rationale="test",
        finalists_for_5c4c=(),
        warnings=(),
        resources_touched=(),
    )
    payload = arena_report_to_json(report)
    assert "5c4b-v1" in payload

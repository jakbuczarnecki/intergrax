"""VPI embedding model arena orchestration."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_execution_configuration import (
    VpiEmbeddingDeviceUnavailableError,
    assert_execution_device_available,
    load_vpi_embedding_provider_execution_configuration,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.config import (
    load_vpi_embedding_materialization_config,
)
from platform_proofs.scenarios.verified_product_identification.arena.composition.candidates import (
    BASELINE_CANDIDATE_ID,
    BASELINE_KNOWN_THROUGHPUT_RPS,
    DEFAULT_BATCH_CANDIDATES,
    DEFAULT_STAGE_A_RECORDS,
    DEFAULT_STAGE_B_RECORDS,
    DEFAULT_STAGE_C_RECORDS,
    FULL_DATASET_RECORD_COUNT,
    build_default_arena_candidates,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.candidates import (
    EmbeddingArenaCandidate,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.classification import (
    EmbeddingArenaStageStatus,
    EmbeddingArenaVerdict,
    EmbeddingLicenseClassification,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.query_benchmark import (
    EmbeddingArenaQueryCase,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.results import (
    ArenaSampleManifest,
    CandidateArenaResult,
    CandidateRuntimeMetadata,
    CandidateStageSnapshot,
    EmbeddingArenaReport,
    QueryLatencySnapshot,
    RetrievalQualityMetrics,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.versioning import (
    ARENA_QUERY_BENCHMARK_VERSION,
    VPI_EMBEDDING_ARENA_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.estimates import (
    estimate_artifact_size,
    estimate_preliminary_full_build,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.metrics import (
    aggregate_retrieval_metrics,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.query_builder import (
    build_query_benchmark_cases,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.search import (
    rank_corpus_by_cosine_similarity,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.verdict import (
    classify_candidate_verdict,
    compute_quality_delta,
    compute_speedup_estimate,
    decide_arena_outcome,
)
from platform_proofs.scenarios.verified_product_identification.arena.integration.embedding_execution import (
    embed_documents,
    embed_query_vector,
    measure_candidate_warmup,
    measure_query_latency,
    run_candidate_microbenchmark,
)
from platform_proofs.scenarios.verified_product_identification.arena.integration.truncation_probe import (
    profile_truncation_for_texts,
)
from platform_proofs.scenarios.verified_product_identification.arena.sampling.arena_sample import (
    ArenaSampleRecord,
    build_arena_sample_manifest,
)
from platform_proofs.scenarios.verified_product_identification.arena.sampling.dataset_loader import (
    load_arena_sample_records,
)
from platform_proofs.scenarios.verified_product_identification.qualification.batch_selection import (
    select_best_provider_batch_size,
)
from platform_proofs.scenarios.verified_product_identification.qualification.contracts.results import (
    MicrobenchmarkCandidateStatus,
)
from platform_proofs.scenarios.verified_product_identification.qualification.integration.hardware_probe import (
    probe_hardware_runtime_capability,
)
from platform_proofs.scenarios.verified_product_identification.qualification.text_length_profile import (
    profile_text_lengths,
)


def _offer_id_to_index(records: Sequence[ArenaSampleRecord]) -> dict[str, int]:
    return {record.offer_id: index for index, record in enumerate(records)}


def _evaluate_retrieval_quality(
    *,
    corpus_embeddings: np.ndarray,
    query_cases: Sequence[EmbeddingArenaQueryCase],
    offer_index: dict[str, int],
    embed_query,
) -> RetrievalQualityMetrics:
    per_query_relevant: list[list[int]] = []
    per_query_ranked: list[list[int]] = []
    for case in query_cases:
        relevant = [
            offer_index[source_ref.offer_id]
            for source_ref in case.relevant_source_refs
            if source_ref.offer_id in offer_index
        ]
        if not relevant:
            continue
        query_vector = embed_query(case.query_text)
        ranked = rank_corpus_by_cosine_similarity(
            corpus_embeddings,
            query_vector,
            top_k=min(10, corpus_embeddings.shape[0]),
        )
        per_query_relevant.append(relevant)
        per_query_ranked.append(list(ranked))
    return aggregate_retrieval_metrics(per_query_relevant, per_query_ranked)


def _stage_status_from_microbenchmark(results) -> EmbeddingArenaStageStatus:
    passing = [
        item
        for item in results
        if item.status is MicrobenchmarkCandidateStatus.PASS and item.records_per_second > 0.0
    ]
    if passing:
        return EmbeddingArenaStageStatus.PASS
    if any(item.status is MicrobenchmarkCandidateStatus.FAILED_OOM for item in results):
        return EmbeddingArenaStageStatus.FAILED_OOM
    return EmbeddingArenaStageStatus.FAILED_RUNTIME


def _run_stage(
    candidate: EmbeddingArenaCandidate,
    records: Sequence[ArenaSampleRecord],
    *,
    stage_name: str,
    batch_candidates: tuple[int, ...],
    device: str | None,
) -> CandidateStageSnapshot:
    texts = tuple(record.semantic_text for record in records)
    if candidate.fixed_provider_batch_size is not None:
        batch_size = candidate.fixed_provider_batch_size
        microbenchmark_results = (
            run_candidate_microbenchmark(
                candidate,
                texts,
                provider_batch_size=batch_size,
                device=device,
            ),
        )
        selection_rationale = (
            f"reused known baseline batch size {batch_size} from 5C4A2 evidence"
        )
    else:
        microbenchmark_results = tuple(
            run_candidate_microbenchmark(
                candidate,
                texts,
                provider_batch_size=batch_candidate,
                device=device,
            )
            for batch_candidate in batch_candidates
        )
        batch_size, selection_rationale = select_best_provider_batch_size(
            microbenchmark_results,
            expected_dimension=candidate.expected_dimension,
        )
        if batch_size is None:
            return CandidateStageSnapshot(
                stage_name=stage_name,
                record_count=len(records),
                status=_stage_status_from_microbenchmark(microbenchmark_results),
                selected_provider_batch_size=None,
                warmup_timing=None,
                microbenchmark_results=microbenchmark_results,
                throughput_records_per_second=None,
                peak_vram_bytes=None,
                output_dimension=None,
                detail="no stable provider batch candidate",
            )

    warmup_timing = measure_candidate_warmup(
        candidate,
        texts,
        provider_batch_size=batch_size,
        device=device,
    )
    best = next(
        (
            item
            for item in microbenchmark_results
            if item.provider_batch_size == batch_size
            and item.status is MicrobenchmarkCandidateStatus.PASS
        ),
        microbenchmark_results[0],
    )
    return CandidateStageSnapshot(
        stage_name=stage_name,
        record_count=len(records),
        status=_stage_status_from_microbenchmark(microbenchmark_results),
        selected_provider_batch_size=batch_size,
        warmup_timing=warmup_timing,
        microbenchmark_results=microbenchmark_results,
        throughput_records_per_second=best.records_per_second,
        peak_vram_bytes=best.peak_vram_bytes,
        output_dimension=candidate.expected_dimension,
        detail=selection_rationale,
    )


def _runtime_metadata(
    candidate: EmbeddingArenaCandidate,
    *,
    batch_size: int | None,
    device: str | None,
) -> CandidateRuntimeMetadata:
    sentence_transformers_version: str | None = None
    transformers_version: str | None = None
    torch_version: str | None = None
    try:
        import sentence_transformers

        sentence_transformers_version = sentence_transformers.__version__
    except ImportError:
        pass
    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:
        pass
    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        pass
    return CandidateRuntimeMetadata(
        provider=candidate.provider,
        model=candidate.model,
        resolved_revision=None,
        dimension=candidate.expected_dimension,
        input_policy_version=candidate.semantic_input_policy_id,
        normalization="provider_expected" if candidate.normalization_expected else "unknown",
        dtype="provider_default",
        device=device,
        batch_size=batch_size,
        sentence_transformers_version=sentence_transformers_version,
        transformers_version=transformers_version,
        torch_version=torch_version,
        trust_remote_code_required=candidate.trust_remote_code_required,
        requires_remote_code=candidate.trust_remote_code_required,
    )


def run_embedding_arena(
    *,
    include_e5_control: bool = False,
    run_gpu_stages: bool = True,
    session_dir: str | None = None,
) -> EmbeddingArenaReport:
    warnings: list[str] = []
    resources_touched: list[str] = []
    if session_dir is not None:
        resources_touched.append(session_dir)

    materialization_config = load_vpi_embedding_materialization_config()
    records = load_arena_sample_records(materialization_config, target_size=DEFAULT_STAGE_C_RECORDS)
    sample_manifest = build_arena_sample_manifest(records)
    query_cases = build_query_benchmark_cases(records)
    text_length_profile = profile_text_lengths(tuple(record.semantic_text for record in records))

    execution_configuration = load_vpi_embedding_provider_execution_configuration()
    hardware = None
    device: str | None = execution_configuration.device
    gpu_available = True
    try:
        assert_execution_device_available(execution_configuration)
    except VpiEmbeddingDeviceUnavailableError as exc:
        gpu_available = False
        run_gpu_stages = False
        warnings.append(str(exc))

    if gpu_available:
        hardware = probe_hardware_runtime_capability(
            configured_device=execution_configuration.device,
        )

    candidates = build_default_arena_candidates(include_e5_control=include_e5_control)
    offer_index = _offer_id_to_index(records)
    candidate_results: list[CandidateArenaResult] = []
    baseline_quality: RetrievalQualityMetrics | None = None
    baseline_throughput = BASELINE_KNOWN_THROUGHPUT_RPS
    baseline_embedding_hours: float | None = None
    stage_b_rankings: list[tuple[str, float, RetrievalQualityMetrics | None]] = []

    for candidate in candidates:
        candidate_warnings: list[str] = []
        if candidate.license_classification is EmbeddingLicenseClassification.REJECTED:
            candidate_results.append(
                CandidateArenaResult(
                    candidate_id=candidate.candidate_id,
                    verdict=EmbeddingArenaVerdict.REJECTED_LICENSE,
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
                    warnings=tuple(candidate_warnings),
                )
            )
            continue

        truncation_profile = None
        if candidate.max_sequence_length is not None:
            try:
                truncation_profile = profile_truncation_for_texts(
                    model_name=candidate.model,
                    texts=tuple(record.semantic_text for record in records),
                    max_supported_tokens=candidate.max_sequence_length,
                )
            except Exception as exc:
                candidate_warnings.append(f"truncation profiling unavailable: {exc}")

        stage_a = stage_b = stage_c = None
        quality_metrics = None
        long_input_quality_metrics = None
        quality_delta = None
        query_latency = None
        artifact_size_estimate = None
        full_build_estimate = None
        speedup_estimate = None
        runtime_ok = True
        correctness_ok = True

        if run_gpu_stages and gpu_available:
            stage_a = _run_stage(
                candidate,
                records[:DEFAULT_STAGE_A_RECORDS],
                stage_name="stage_a",
                batch_candidates=DEFAULT_BATCH_CANDIDATES,
                device=device,
            )
            if stage_a.status is not EmbeddingArenaStageStatus.PASS:
                runtime_ok = False
            else:
                stage_b = _run_stage(
                    candidate,
                    records[:DEFAULT_STAGE_B_RECORDS],
                    stage_name="stage_b",
                    batch_candidates=DEFAULT_BATCH_CANDIDATES,
                    device=device,
                )
                if stage_b.status is not EmbeddingArenaStageStatus.PASS:
                    runtime_ok = False
                elif stage_b.selected_provider_batch_size is not None:
                    batch_size = stage_b.selected_provider_batch_size
                    corpus = embed_documents(
                        candidate,
                        canonical_texts=tuple(record.semantic_text for record in records[:DEFAULT_STAGE_B_RECORDS]),
                        provider_batch_size=batch_size,
                        device=device,
                    )

                    def _embed_query(query_text: str) -> np.ndarray:
                        return embed_query_vector(
                            candidate,
                            query_text=query_text,
                            provider_batch_size=batch_size,
                            device=device,
                        )

                    quality_metrics = _evaluate_retrieval_quality(
                        corpus_embeddings=corpus,
                        query_cases=query_cases,
                        offer_index=offer_index,
                        embed_query=_embed_query,
                    )
                    long_cases = tuple(case for case in query_cases if case.is_long_input_query)
                    if long_cases:
                        long_input_quality_metrics = _evaluate_retrieval_quality(
                            corpus_embeddings=corpus,
                            query_cases=long_cases,
                            offer_index=offer_index,
                            embed_query=_embed_query,
                        )
                    stage_b_rankings.append(
                        (
                            candidate.candidate_id,
                            stage_b.throughput_records_per_second or 0.0,
                            quality_metrics,
                        )
                    )
        else:
            candidate_warnings.append("GPU stages skipped; throughput/quality evidence unavailable")

        promote_to_stage_c = candidate.is_baseline
        if not candidate.is_baseline and quality_metrics is not None:
            promote_to_stage_c = True
        if (
            not candidate.is_baseline
            and stage_b is not None
            and stage_b.throughput_records_per_second is not None
            and stage_b.throughput_records_per_second >= baseline_throughput * 1.2
        ):
            promote_to_stage_c = True

        finalists = sorted(
            stage_b_rankings,
            key=lambda item: (item[2].recall_at_10 if item[2] is not None else 0.0, item[1]),
            reverse=True,
        )
        top_finalist_ids = {item[0] for item in finalists[:3]}
        if (
            run_gpu_stages
            and gpu_available
            and promote_to_stage_c
            and (
                candidate.is_baseline
                or candidate.candidate_id in top_finalist_ids
                or len(stage_b_rankings) <= 3
            )
        ):
            stage_c = _run_stage(
                candidate,
                records[:DEFAULT_STAGE_C_RECORDS],
                stage_name="stage_c",
                batch_candidates=DEFAULT_BATCH_CANDIDATES,
                device=device,
            )
            if stage_c.status is EmbeddingArenaStageStatus.PASS and stage_c.selected_provider_batch_size is not None:
                batch_size = stage_c.selected_provider_batch_size
                corpus = embed_documents(
                    candidate,
                    canonical_texts=tuple(record.semantic_text for record in records[:DEFAULT_STAGE_C_RECORDS]),
                    provider_batch_size=batch_size,
                    device=device,
                )

                def _embed_query_stage_c(query_text: str) -> np.ndarray:
                    return embed_query_vector(
                        candidate,
                        query_text=query_text,
                        provider_batch_size=batch_size,
                        device=device,
                    )

                quality_metrics = _evaluate_retrieval_quality(
                    corpus_embeddings=corpus,
                    query_cases=query_cases,
                    offer_index=offer_index,
                    embed_query=_embed_query_stage_c,
                )
                long_cases = tuple(case for case in query_cases if case.is_long_input_query)
                if long_cases:
                    long_input_quality_metrics = _evaluate_retrieval_quality(
                        corpus_embeddings=corpus,
                        query_cases=long_cases,
                        offer_index=offer_index,
                        embed_query=_embed_query_stage_c,
                    )
                p50, p95 = measure_query_latency(
                    candidate,
                    query_texts=tuple(case.query_text for case in query_cases[:5]),
                    provider_batch_size=batch_size,
                    device=device,
                )
                query_latency = QueryLatencySnapshot(
                    single_query_p50_seconds=p50,
                    single_query_p95_seconds=p95,
                    small_batch_records_per_second=stage_c.throughput_records_per_second,
                )
                artifact_size_estimate = estimate_artifact_size(
                    dimension=candidate.expected_dimension,
                    record_count=FULL_DATASET_RECORD_COUNT,
                )
                throughput = stage_c.throughput_records_per_second or baseline_throughput
                full_build_estimate = estimate_preliminary_full_build(
                    record_count=FULL_DATASET_RECORD_COUNT,
                    steady_records_per_second=throughput,
                    throughput_source="arena_stage_c",
                )
                if candidate.candidate_id == BASELINE_CANDIDATE_ID:
                    baseline_throughput = throughput
                    baseline_quality = quality_metrics
                    baseline_embedding_hours = full_build_estimate.estimated_embedding_hours
                elif baseline_throughput > 0.0:
                    speedup_estimate = compute_speedup_estimate(
                        candidate_records_per_second=throughput,
                        baseline_records_per_second=baseline_throughput,
                        baseline_embedding_hours=baseline_embedding_hours,
                    )

        if (
            quality_metrics is not None
            and baseline_quality is not None
            and not candidate.is_baseline
        ):
            quality_delta = compute_quality_delta(quality_metrics, baseline_quality)

        long_input_regression = False
        if (
            long_input_quality_metrics is not None
            and baseline_quality is not None
            and truncation_profile is not None
            and truncation_profile.truncated_percentage > 0.0
        ):
            long_input_regression = (
                long_input_quality_metrics.recall_at_10
                < baseline_quality.recall_at_10 - 0.10
            )

        verdict = classify_candidate_verdict(
            is_baseline=candidate.is_baseline,
            license_eligible=candidate.license_classification
            is EmbeddingLicenseClassification.ELIGIBLE_COMMERCIAL,
            runtime_ok=runtime_ok,
            correctness_ok=correctness_ok,
            quality_delta=quality_delta,
            speedup=speedup_estimate,
            long_input_regression=long_input_regression,
        )

        runtime_metadata = _runtime_metadata(
            candidate,
            batch_size=(
                stage_c.selected_provider_batch_size
                if stage_c is not None
                else stage_b.selected_provider_batch_size
                if stage_b is not None
                else candidate.fixed_provider_batch_size
            ),
            device=device,
        )

        candidate_results.append(
            CandidateArenaResult(
                candidate_id=candidate.candidate_id,
                verdict=verdict,
                runtime_metadata=runtime_metadata,
                truncation_profile=truncation_profile,
                stage_a=stage_a,
                stage_b=stage_b,
                stage_c=stage_c,
                quality_metrics=quality_metrics,
                long_input_quality_metrics=long_input_quality_metrics,
                quality_delta_vs_baseline=quality_delta,
                query_latency=query_latency,
                artifact_size_estimate=artifact_size_estimate,
                full_build_estimate=full_build_estimate,
                speedup_estimate=speedup_estimate,
                warnings=tuple(candidate_warnings),
            )
        )

    decision, rationale, finalists = decide_arena_outcome(tuple(candidate_results))
    return EmbeddingArenaReport(
        arena_version=VPI_EMBEDDING_ARENA_VERSION,
        sample_manifest=sample_manifest,
        query_benchmark_version=ARENA_QUERY_BENCHMARK_VERSION,
        query_cases=query_cases,
        hardware=hardware,
        text_length_profile=text_length_profile,
        candidate_results=tuple(candidate_results),
        decision=decision,
        decision_rationale=rationale,
        finalists_for_5c4c=finalists,
        warnings=tuple(warnings),
        resources_touched=tuple(resources_touched),
    )

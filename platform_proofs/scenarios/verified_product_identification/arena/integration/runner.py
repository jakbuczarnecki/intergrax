"""VPI embedding model arena orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
from platform_proofs.scenarios.verified_product_identification.arena.contracts.errors import (
    EmbeddingArenaTokenizerUnavailableError,
    EmbeddingArenaTruncationProfileError,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.results import (
    ArtifactSizeEstimate,
    CandidateArenaResult,
    CandidateRuntimeMetadata,
    CandidateStageSnapshot,
    EmbeddingArenaReport,
    FullBuildEstimate,
    QualityDeltaMetrics,
    QueryLatencySnapshot,
    RetrievalQualityMetrics,
    SpeedupEstimate,
    TruncationProfile,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.stage_evaluation_scope import (
    EmbeddingArenaStageEvaluationScope,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.versioning import (
    ARENA_QUERY_BENCHMARK_VERSION,
    VPI_EMBEDDING_ARENA_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.estimates import (
    estimate_artifact_size,
    estimate_preliminary_full_build,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.finalist_selection import (
    StageBCandidateEvidence,
    select_stage_c_finalist_ids,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.retrieval_evaluation import (
    evaluate_retrieval_quality_for_scope,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.stage_scope import (
    build_stage_evaluation_scope,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.verdict import (
    classify_candidate_verdict,
    compute_quality_delta,
    compute_speedup_estimate,
    decide_arena_outcome,
)
from platform_proofs.scenarios.verified_product_identification.arena.integration.candidate_execution_session import (
    EmbeddingArenaCandidateExecutionSession,
)
from platform_proofs.scenarios.verified_product_identification.arena.integration.embedding_execution import (
    measure_candidate_warmup,
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


@dataclass(frozen=True, slots=True)
class _CandidateStageWork:
    candidate: EmbeddingArenaCandidate
    warnings: tuple[str, ...]
    truncation_profile: TruncationProfile | None
    truncation_ok: bool
    stage_a: CandidateStageSnapshot | None
    stage_b: CandidateStageSnapshot | None
    stage_b_quality: RetrievalQualityMetrics | None
    stage_b_long_input_quality: RetrievalQualityMetrics | None
    runtime_ok: bool


@dataclass(frozen=True, slots=True)
class _CandidateFinalWork:
    stage_c: CandidateStageSnapshot | None
    quality_metrics: RetrievalQualityMetrics | None
    long_input_quality_metrics: RetrievalQualityMetrics | None
    quality_delta: QualityDeltaMetrics | None
    query_latency: QueryLatencySnapshot | None
    artifact_size_estimate: ArtifactSizeEstimate | None
    full_build_estimate: FullBuildEstimate | None
    speedup_estimate: SpeedupEstimate | None
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


def _evaluate_stage_quality(
    candidate: EmbeddingArenaCandidate,
    scope: EmbeddingArenaStageEvaluationScope,
    *,
    batch_size: int,
    device: str | None,
) -> tuple[RetrievalQualityMetrics, RetrievalQualityMetrics | None]:
    canonical_texts = tuple(record.semantic_text for record in scope.records)
    with EmbeddingArenaCandidateExecutionSession(
        candidate,
        provider_batch_size=batch_size,
        device=device,
    ) as session:
        session.warmup(canonical_texts[: min(8, len(canonical_texts))])
        corpus = session.embed_documents(
            canonical_texts,
            expected_dimension=candidate.expected_dimension,
        )
        query_texts = tuple(case.query_text for case in scope.query_cases)
        query_embeddings = session.embed_queries(
            query_texts,
            expected_dimension=candidate.expected_dimension,
        )
        quality_metrics = evaluate_retrieval_quality_for_scope(
            scope=scope,
            corpus_embeddings=corpus,
            query_embeddings=query_embeddings,
            expected_dimension=candidate.expected_dimension,
        )
        long_cases = tuple(case for case in scope.query_cases if case.is_long_input_query)
        long_input_quality_metrics = None
        if long_cases:
            long_scope_cases = scope.query_cases
            long_query_texts = tuple(case.query_text for case in long_cases)
            long_indices = [
                index
                for index, case in enumerate(long_scope_cases)
                if case.is_long_input_query
            ]
            long_query_embeddings = query_embeddings[long_indices, :]
            long_scope = EmbeddingArenaStageEvaluationScope(
                stage_name=scope.stage_name,
                records=scope.records,
                query_cases=long_cases,
                offer_index=scope.offer_index,
                corpus_size=scope.corpus_size,
                benchmark_version=scope.benchmark_version,
                sample_version=scope.sample_version,
                content_fingerprint=scope.content_fingerprint,
            )
            long_input_quality_metrics = evaluate_retrieval_quality_for_scope(
                scope=long_scope,
                corpus_embeddings=corpus,
                query_embeddings=long_query_embeddings,
                expected_dimension=candidate.expected_dimension,
            )
    return quality_metrics, long_input_quality_metrics


def _profile_truncation(
    candidate: EmbeddingArenaCandidate,
    records: tuple[ArenaSampleRecord, ...],
) -> tuple[TruncationProfile | None, bool, list[str]]:
    warnings: list[str] = []
    if candidate.max_sequence_length is None:
        return None, True, warnings
    try:
        profile = profile_truncation_for_texts(
            model_name=candidate.model,
            texts=tuple(record.semantic_text for record in records),
            max_supported_tokens=candidate.max_sequence_length,
        )
    except EmbeddingArenaTokenizerUnavailableError as exc:
        warnings.append(f"truncation profiling unavailable: {exc}")
        return None, False, warnings
    except EmbeddingArenaTruncationProfileError as exc:
        warnings.append(f"truncation profiling failed: {exc}")
        return None, False, warnings
    return profile, True, warnings


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


def _run_stage_ab_for_candidate(
    candidate: EmbeddingArenaCandidate,
    *,
    records: tuple[ArenaSampleRecord, ...],
    stage_b_scope: EmbeddingArenaStageEvaluationScope,
    run_gpu_stages: bool,
    gpu_available: bool,
    device: str | None,
) -> _CandidateStageWork:
    candidate_warnings: list[str] = []
    truncation_profile, truncation_ok, truncation_warnings = _profile_truncation(
        candidate,
        records,
    )
    candidate_warnings.extend(truncation_warnings)

    stage_a = stage_b = None
    stage_b_quality = None
    stage_b_long_input_quality = None
    runtime_ok = truncation_ok

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
                stage_b_quality, stage_b_long_input_quality = _evaluate_stage_quality(
                    candidate,
                    stage_b_scope,
                    batch_size=stage_b.selected_provider_batch_size,
                    device=device,
                )
    else:
        candidate_warnings.append("GPU stages skipped; throughput/quality evidence unavailable")
        runtime_ok = False

    return _CandidateStageWork(
        candidate=candidate,
        warnings=tuple(candidate_warnings),
        truncation_profile=truncation_profile,
        truncation_ok=truncation_ok,
        stage_a=stage_a,
        stage_b=stage_b,
        stage_b_quality=stage_b_quality,
        stage_b_long_input_quality=stage_b_long_input_quality,
        runtime_ok=runtime_ok,
    )


def _run_stage_c_for_candidate(
    candidate: EmbeddingArenaCandidate,
    *,
    records: tuple[ArenaSampleRecord, ...],
    stage_c_scope: EmbeddingArenaStageEvaluationScope,
    device: str | None,
    baseline_throughput: float,
    baseline_embedding_hours: float | None,
) -> _CandidateFinalWork:
    stage_c = _run_stage(
        candidate,
        records[:DEFAULT_STAGE_C_RECORDS],
        stage_name="stage_c",
        batch_candidates=DEFAULT_BATCH_CANDIDATES,
        device=device,
    )
    quality_metrics = None
    long_input_quality_metrics = None
    quality_delta = None
    query_latency = None
    artifact_size_estimate = None
    full_build_estimate = None
    speedup_estimate = None

    if stage_c.status is EmbeddingArenaStageStatus.PASS and stage_c.selected_provider_batch_size is not None:
        batch_size = stage_c.selected_provider_batch_size
        quality_metrics, long_input_quality_metrics = _evaluate_stage_quality(
            candidate,
            stage_c_scope,
            batch_size=batch_size,
            device=device,
        )
        canonical_texts = tuple(record.semantic_text for record in stage_c_scope.records)
        with EmbeddingArenaCandidateExecutionSession(
            candidate,
            provider_batch_size=batch_size,
            device=device,
        ) as session:
            session.warmup(canonical_texts[: min(8, len(canonical_texts))])
            p50, p95 = session.measure_query_latency(
                tuple(case.query_text for case in stage_c_scope.query_cases[:5]),
                expected_dimension=candidate.expected_dimension,
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
        if candidate.candidate_id != BASELINE_CANDIDATE_ID and baseline_throughput > 0.0:
            speedup_estimate = compute_speedup_estimate(
                candidate_records_per_second=throughput,
                baseline_records_per_second=baseline_throughput,
                baseline_embedding_hours=baseline_embedding_hours,
            )

    return _CandidateFinalWork(
        stage_c=stage_c,
        quality_metrics=quality_metrics,
        long_input_quality_metrics=long_input_quality_metrics,
        quality_delta=quality_delta,
        query_latency=query_latency,
        artifact_size_estimate=artifact_size_estimate,
        full_build_estimate=full_build_estimate,
        speedup_estimate=speedup_estimate,
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
    stage_b_scope = build_stage_evaluation_scope(
        stage_name="stage_b",
        records=tuple(records[:DEFAULT_STAGE_B_RECORDS]),
    )
    stage_c_scope = build_stage_evaluation_scope(
        stage_name="stage_c",
        records=tuple(records[:DEFAULT_STAGE_C_RECORDS]),
    )
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
    baseline_throughput = BASELINE_KNOWN_THROUGHPUT_RPS
    baseline_quality: RetrievalQualityMetrics | None = None
    baseline_embedding_hours: float | None = None

    stage_work: dict[str, _CandidateStageWork] = {}
    for candidate in candidates:
        if candidate.license_classification is EmbeddingLicenseClassification.REJECTED:
            stage_work[candidate.candidate_id] = _CandidateStageWork(
                candidate=candidate,
                warnings=(),
                truncation_profile=None,
                truncation_ok=True,
                stage_a=None,
                stage_b=None,
                stage_b_quality=None,
                stage_b_long_input_quality=None,
                runtime_ok=False,
            )
            continue
        stage_work[candidate.candidate_id] = _run_stage_ab_for_candidate(
            candidate,
            records=tuple(records),
            stage_b_scope=stage_b_scope,
            run_gpu_stages=run_gpu_stages,
            gpu_available=gpu_available,
            device=device,
        )

    stage_b_evidence = tuple(
        StageBCandidateEvidence(
            candidate_id=work.candidate.candidate_id,
            is_baseline=work.candidate.is_baseline,
            license_eligible=work.candidate.license_classification
            is EmbeddingLicenseClassification.ELIGIBLE_COMMERCIAL,
            runtime_ok=work.runtime_ok
            and work.stage_b is not None
            and work.stage_b.status is EmbeddingArenaStageStatus.PASS,
            throughput_records_per_second=(
                work.stage_b.throughput_records_per_second if work.stage_b is not None else None
            ),
            quality_metrics=work.stage_b_quality,
        )
        for work in stage_work.values()
        if work.candidate.license_classification is not EmbeddingLicenseClassification.REJECTED
    )
    finalist_ids = select_stage_c_finalist_ids(
        stage_b_evidence,
        baseline_candidate_id=BASELINE_CANDIDATE_ID,
        baseline_throughput=baseline_throughput,
    )

    final_work: dict[str, _CandidateFinalWork] = {}
    if run_gpu_stages and gpu_available:
        ordered_finalist_ids = sorted(finalist_ids)
        if BASELINE_CANDIDATE_ID in finalist_ids:
            ordered_finalist_ids = (BASELINE_CANDIDATE_ID,) + tuple(
                finalist_id
                for finalist_id in ordered_finalist_ids
                if finalist_id != BASELINE_CANDIDATE_ID
            )
        for finalist_id in ordered_finalist_ids:
            work = stage_work.get(finalist_id)
            if work is None:
                continue
            final_work[finalist_id] = _run_stage_c_for_candidate(
                work.candidate,
                records=tuple(records),
                stage_c_scope=stage_c_scope,
                device=device,
                baseline_throughput=baseline_throughput,
                baseline_embedding_hours=baseline_embedding_hours,
            )
            if finalist_id == BASELINE_CANDIDATE_ID:
                final = final_work[finalist_id]
                if final.quality_metrics is not None:
                    baseline_quality = final.quality_metrics
                if final.full_build_estimate is not None:
                    baseline_embedding_hours = final.full_build_estimate.estimated_embedding_hours
                if (
                    final.stage_c is not None
                    and final.stage_c.throughput_records_per_second is not None
                ):
                    baseline_throughput = final.stage_c.throughput_records_per_second

        if baseline_throughput > 0.0 and baseline_embedding_hours is not None:
            for finalist_id, final in final_work.items():
                if finalist_id == BASELINE_CANDIDATE_ID:
                    continue
                if final.stage_c is None or final.stage_c.throughput_records_per_second is None:
                    continue
                final_work[finalist_id] = _CandidateFinalWork(
                    stage_c=final.stage_c,
                    quality_metrics=final.quality_metrics,
                    long_input_quality_metrics=final.long_input_quality_metrics,
                    quality_delta=final.quality_delta,
                    query_latency=final.query_latency,
                    artifact_size_estimate=final.artifact_size_estimate,
                    full_build_estimate=final.full_build_estimate,
                    speedup_estimate=compute_speedup_estimate(
                        candidate_records_per_second=final.stage_c.throughput_records_per_second,
                        baseline_records_per_second=baseline_throughput,
                        baseline_embedding_hours=baseline_embedding_hours,
                    ),
                )

    candidate_results: list[CandidateArenaResult] = []
    for candidate in candidates:
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
                    warnings=(),
                )
            )
            continue

        work = stage_work[candidate.candidate_id]
        final = final_work.get(candidate.candidate_id)
        quality_metrics = final.quality_metrics if final is not None else work.stage_b_quality
        long_input_quality_metrics = (
            final.long_input_quality_metrics
            if final is not None
            else work.stage_b_long_input_quality
        )
        quality_delta = None
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
            and work.truncation_profile is not None
            and work.truncation_profile.truncated_percentage > 0.0
        ):
            long_input_regression = (
                long_input_quality_metrics.recall_at_10
                < baseline_quality.recall_at_10 - 0.10
            )

        correctness_ok = work.truncation_ok
        verdict = classify_candidate_verdict(
            is_baseline=candidate.is_baseline,
            license_eligible=candidate.license_classification
            is EmbeddingLicenseClassification.ELIGIBLE_COMMERCIAL,
            runtime_ok=work.runtime_ok,
            correctness_ok=correctness_ok,
            quality_delta=quality_delta,
            speedup=final.speedup_estimate if final is not None else None,
            long_input_regression=long_input_regression,
        )

        runtime_metadata = _runtime_metadata(
            candidate,
            batch_size=(
                final.stage_c.selected_provider_batch_size
                if final is not None and final.stage_c is not None
                else work.stage_b.selected_provider_batch_size
                if work.stage_b is not None
                else candidate.fixed_provider_batch_size
            ),
            device=device,
        )

        candidate_results.append(
            CandidateArenaResult(
                candidate_id=candidate.candidate_id,
                verdict=verdict,
                runtime_metadata=runtime_metadata,
                truncation_profile=work.truncation_profile,
                stage_a=work.stage_a,
                stage_b=work.stage_b,
                stage_c=final.stage_c if final is not None else None,
                quality_metrics=quality_metrics,
                long_input_quality_metrics=long_input_quality_metrics,
                quality_delta_vs_baseline=quality_delta,
                query_latency=final.query_latency if final is not None else None,
                artifact_size_estimate=final.artifact_size_estimate if final is not None else None,
                full_build_estimate=final.full_build_estimate if final is not None else None,
                speedup_estimate=final.speedup_estimate if final is not None else None,
                warnings=work.warnings,
            )
        )

    decision, rationale, finalists = decide_arena_outcome(tuple(candidate_results))
    return EmbeddingArenaReport(
        arena_version=VPI_EMBEDDING_ARENA_VERSION,
        sample_manifest=sample_manifest,
        query_benchmark_version=ARENA_QUERY_BENCHMARK_VERSION,
        query_cases=stage_c_scope.query_cases,
        hardware=hardware,
        text_length_profile=text_length_profile,
        candidate_results=tuple(candidate_results),
        decision=decision,
        decision_rationale=rationale,
        finalists_for_5c4c=finalists,
        warnings=tuple(warnings),
        resources_touched=tuple(resources_touched),
    )

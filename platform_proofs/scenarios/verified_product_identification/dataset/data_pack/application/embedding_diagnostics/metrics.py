"""Statistical aggregation and bottleneck classification for embedding diagnostics."""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.contracts import (
    BATCH_EXPERIMENT_SIZES,
    DIAGNOSTIC_MAX_RECORD_LIMIT,
    EmbeddingBottleneckCase,
    EmbeddingDiagnosticClassification,
    EmbeddingPerformanceMetrics,
    LOW_GPU_UTILIZATION_PERCENT,
    LOW_THROUGHPUT_RECORDS_PER_SECOND,
    RecordRepresentationMeasurement,
    SIGNIFICANT_BATCH_IMPROVEMENT_RATIO,
    TOKEN_P95_REASONABLE_THRESHOLD,
    TOKEN_P95_REPRESENTATION_THRESHOLD,
    TokenDistributionReport,
)


def percentile(values: Sequence[float], percentile_rank: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile_rank
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def validate_record_limit(record_limit: int) -> int:
    if record_limit <= 0:
        msg = "record_limit must be > 0"
        raise ValueError(msg)
    if record_limit > DIAGNOSTIC_MAX_RECORD_LIMIT:
        msg = f"record_limit must be <= {DIAGNOSTIC_MAX_RECORD_LIMIT}"
        raise ValueError(msg)
    return record_limit


def build_token_distribution_report(
    measurements: Sequence[RecordRepresentationMeasurement],
) -> TokenDistributionReport:
    if not measurements:
        msg = "measurements must not be empty"
        raise ValueError(msg)
    token_counts = [float(measurement.token_count) for measurement in measurements]
    semantic_lengths = [float(measurement.semantic_text_length) for measurement in measurements]
    total_tokens = sum(measurement.token_count for measurement in measurements)
    return TokenDistributionReport(
        record_count=len(measurements),
        total_tokens=total_tokens,
        average_tokens=statistics.fmean(token_counts),
        p50_tokens=percentile(token_counts, 0.50),
        p95_tokens=percentile(token_counts, 0.95),
        p99_tokens=percentile(token_counts, 0.99),
        max_tokens=max(measurement.token_count for measurement in measurements),
        semantic_text_length_avg=statistics.fmean(semantic_lengths),
        semantic_text_length_p95=percentile(semantic_lengths, 0.95),
        record_measurements=tuple(measurements),
    )


def build_embedding_performance_metrics(
    *,
    model_id: str,
    model_revision: str,
    provider: str,
    device: str,
    token_distribution: TokenDistributionReport,
    batch_size: int,
    batches_count: int,
    embedding_seconds: float,
    gpu_name: str | None,
    cuda_available: bool,
    peak_memory_mb: float | None,
    average_gpu_utilization_percent: float | None,
) -> EmbeddingPerformanceMetrics:
    record_count = token_distribution.record_count
    records_per_second = record_count / embedding_seconds if embedding_seconds > 0 else 0.0
    tokens_per_second = (
        token_distribution.total_tokens / embedding_seconds if embedding_seconds > 0 else 0.0
    )
    return EmbeddingPerformanceMetrics(
        model_id=model_id,
        model_revision=model_revision,
        provider=provider,
        device=device,
        record_count=record_count,
        semantic_text_length_avg=token_distribution.semantic_text_length_avg,
        semantic_text_length_p95=token_distribution.semantic_text_length_p95,
        total_tokens=token_distribution.total_tokens,
        average_tokens=token_distribution.average_tokens,
        p50_tokens=token_distribution.p50_tokens,
        p95_tokens=token_distribution.p95_tokens,
        p99_tokens=token_distribution.p99_tokens,
        max_tokens=token_distribution.max_tokens,
        batch_size=batch_size,
        batches_count=batches_count,
        embedding_seconds=embedding_seconds,
        records_per_second=records_per_second,
        tokens_per_second=tokens_per_second,
        gpu_name=gpu_name,
        cuda_available=cuda_available,
        peak_memory_mb=peak_memory_mb,
        average_gpu_utilization_percent=average_gpu_utilization_percent,
    )


def classify_embedding_bottleneck(
    *,
    token_distribution: TokenDistributionReport,
    baseline: EmbeddingPerformanceMetrics,
    batch_experiments: Sequence[EmbeddingPerformanceMetrics],
) -> EmbeddingDiagnosticClassification:
    if token_distribution.p95_tokens > TOKEN_P95_REPRESENTATION_THRESHOLD:
        return EmbeddingDiagnosticClassification(
            case=EmbeddingBottleneckCase.REPRESENTATION_OPTIMIZATION,
            conclusion=(
                f"token p95={token_distribution.p95_tokens:.0f} exceeds "
                f"{TOKEN_P95_REPRESENTATION_THRESHOLD}; semantic representation is oversized"
            ),
            recommended_next_task="Embedding Representation Optimization",
        )

    throughputs = [experiment.records_per_second for experiment in batch_experiments]
    if throughputs:
        min_throughput = min(throughputs)
        max_throughput = max(throughputs)
        if (
            min_throughput > 0.0
            and max_throughput / min_throughput >= SIGNIFICANT_BATCH_IMPROVEMENT_RATIO
        ):
            best_batch = max(batch_experiments, key=lambda item: item.records_per_second)
            return EmbeddingDiagnosticClassification(
                case=EmbeddingBottleneckCase.BATCH_TUNING,
                conclusion=(
                    f"batch size materially affects throughput "
                    f"({min_throughput:.3f} to {max_throughput:.3f} records/sec); "
                    f"best observed batch_size={best_batch.batch_size}"
                ),
                recommended_next_task="Production Batch Qualification",
            )

    gpu_samples = [
        experiment.average_gpu_utilization_percent
        for experiment in batch_experiments
        if experiment.average_gpu_utilization_percent is not None
    ]
    if gpu_samples and statistics.fmean(gpu_samples) < LOW_GPU_UTILIZATION_PERCENT:
        return EmbeddingDiagnosticClassification(
            case=EmbeddingBottleneckCase.CUDA_PIPELINE,
            conclusion=(
                f"average GPU utilization {statistics.fmean(gpu_samples):.1f}% is below "
                f"{LOW_GPU_UTILIZATION_PERCENT:.0f}%"
            ),
            recommended_next_task="GPU Execution Optimization",
        )

    if (
        token_distribution.p95_tokens < TOKEN_P95_REASONABLE_THRESHOLD
        and baseline.records_per_second < LOW_THROUGHPUT_RECORDS_PER_SECOND
    ):
        return EmbeddingDiagnosticClassification(
            case=EmbeddingBottleneckCase.PROVIDER_OPTIMIZATION,
            conclusion=(
                f"token p95={token_distribution.p95_tokens:.0f} is reasonable but throughput "
                f"{baseline.records_per_second:.3f} records/sec is low"
            ),
            recommended_next_task="Provider Optimization",
        )

    return EmbeddingDiagnosticClassification(
        case=EmbeddingBottleneckCase.UNDETERMINED,
        conclusion="diagnostic evidence does not match a single dominant bottleneck case",
        recommended_next_task="Review full diagnostic report before selecting optimization target",
    )


def batch_sizes_for_experiments(production_batch_size: int) -> tuple[int, ...]:
    sizes = {production_batch_size, *BATCH_EXPERIMENT_SIZES}
    return tuple(sorted(sizes))

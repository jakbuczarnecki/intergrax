"""Typed JSON document models for embedding diagnostic evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.contracts import (
    BatchLatencyMeasurement,
    EmbeddingDiagnosticReport,
    EmbeddingPerformanceMetrics,
    EmbeddingTokenStatistics,
    RecordRepresentationMeasurement,
    TokenDistributionReport,
)


@dataclass(frozen=True, slots=True)
class ClassificationJsonDocument:
    case: str
    conclusion: str
    recommended_next_task: str


@dataclass(frozen=True, slots=True)
class RecordMeasurementJsonDocument:
    global_row_index: int
    semantic_text_length: int
    character_count: int
    token_count: int


@dataclass(frozen=True, slots=True)
class DiagnosticSampleJsonDocument:
    record_id: str
    source_ref: str
    global_row_index: int
    character_count: int
    estimated_tokens: int
    token_percentile_bucket: str


@dataclass(frozen=True, slots=True)
class TokenStatisticsJsonDocument:
    count: int
    minimum: int
    mean: float
    p50: float
    p90: float
    p95: float
    p99: float
    maximum: int


@dataclass(frozen=True, slots=True)
class TokenDistributionJsonDocument:
    record_count: int
    total_tokens: int
    statistics: TokenStatisticsJsonDocument
    average_tokens: float
    p50_tokens: float
    p95_tokens: float
    p99_tokens: float
    max_tokens: int
    semantic_text_length_avg: float
    semantic_text_length_p95: float
    records: tuple[RecordMeasurementJsonDocument, ...]
    diagnostic_samples: tuple[DiagnosticSampleJsonDocument, ...]


@dataclass(frozen=True, slots=True)
class MetricsIdentityJsonDocument:
    model_id: str
    model_revision: str
    provider: str
    device: str


@dataclass(frozen=True, slots=True)
class MetricsDatasetJsonDocument:
    record_count: int
    semantic_text_length_avg: float
    semantic_text_length_p95: float


@dataclass(frozen=True, slots=True)
class MetricsTokensJsonDocument:
    total_tokens: int
    average_tokens: float
    p50_tokens: float
    p95_tokens: float
    p99_tokens: float
    max_tokens: int


@dataclass(frozen=True, slots=True)
class MetricsExecutionJsonDocument:
    batch_size: int
    batches_count: int
    embedding_seconds: float
    records_per_second: float
    tokens_per_second: float


@dataclass(frozen=True, slots=True)
class MetricsHardwareJsonDocument:
    gpu_name: str | None
    cuda_available: bool
    peak_memory_mb: float | None
    average_gpu_utilization_percent: float | None


@dataclass(frozen=True, slots=True)
class MetricsJsonDocument:
    identity: MetricsIdentityJsonDocument
    dataset: MetricsDatasetJsonDocument
    tokens: MetricsTokensJsonDocument
    execution: MetricsExecutionJsonDocument
    hardware: MetricsHardwareJsonDocument


@dataclass(frozen=True, slots=True)
class BatchLatencyJsonDocument:
    batch_index: int
    batch_size: int
    record_count: int
    input_tokens: int
    tokens_processed: int
    batch_latency_seconds: float
    inference_latency_seconds: float


@dataclass(frozen=True, slots=True)
class ExperimentJsonDocument:
    experiment_kind: str
    batch_size: int
    metrics: MetricsJsonDocument
    batch_latencies: tuple[BatchLatencyJsonDocument, ...]


@dataclass(frozen=True, slots=True)
class DiagnosticReportJsonDocument:
    qualification_id: str
    record_limit: int
    token_distribution: TokenDistributionJsonDocument
    baseline: MetricsJsonDocument
    batch_experiments: tuple[ExperimentJsonDocument, ...]
    classification: ClassificationJsonDocument


def build_diagnostic_report_json_document(
    report: EmbeddingDiagnosticReport,
) -> DiagnosticReportJsonDocument:
    return DiagnosticReportJsonDocument(
        qualification_id=report.qualification_id,
        record_limit=report.record_limit,
        token_distribution=_token_distribution_to_document(report.token_distribution),
        baseline=_metrics_to_document(report.baseline),
        batch_experiments=tuple(
            ExperimentJsonDocument(
                experiment_kind=experiment.experiment_kind.value,
                batch_size=experiment.batch_size,
                metrics=_metrics_to_document(experiment.metrics),
                batch_latencies=tuple(
                    _batch_latency_to_document(latency) for latency in experiment.batch_latencies
                ),
            )
            for experiment in report.batch_experiments
        ),
        classification=ClassificationJsonDocument(
            case=report.classification.case.value,
            conclusion=report.classification.conclusion,
            recommended_next_task=report.classification.recommended_next_task,
        ),
    )


def serialize_diagnostic_report_json(report: EmbeddingDiagnosticReport) -> str:
    document = build_diagnostic_report_json_document(report)
    return json.dumps(asdict(document), indent=2)


def _token_statistics_to_document(
    statistics: EmbeddingTokenStatistics,
) -> TokenStatisticsJsonDocument:
    return TokenStatisticsJsonDocument(
        count=statistics.count,
        minimum=statistics.minimum,
        mean=round(statistics.mean, 3),
        p50=round(statistics.p50, 3),
        p90=round(statistics.p90, 3),
        p95=round(statistics.p95, 3),
        p99=round(statistics.p99, 3),
        maximum=statistics.maximum,
    )


def _token_distribution_to_document(
    report: TokenDistributionReport,
) -> TokenDistributionJsonDocument:
    return TokenDistributionJsonDocument(
        record_count=report.record_count,
        total_tokens=report.total_tokens,
        statistics=_token_statistics_to_document(report.statistics),
        average_tokens=round(report.statistics.mean, 3),
        p50_tokens=round(report.statistics.p50, 3),
        p95_tokens=round(report.statistics.p95, 3),
        p99_tokens=round(report.statistics.p99, 3),
        max_tokens=report.statistics.maximum,
        semantic_text_length_avg=round(report.semantic_text_length_avg, 3),
        semantic_text_length_p95=round(report.semantic_text_length_p95, 3),
        records=tuple(
            RecordMeasurementJsonDocument(
                global_row_index=measurement.global_row_index,
                semantic_text_length=measurement.semantic_text_length,
                character_count=measurement.character_count,
                token_count=measurement.token_count,
            )
            for measurement in report.record_measurements
        ),
        diagnostic_samples=tuple(
            DiagnosticSampleJsonDocument(
                record_id=sample.record_id,
                source_ref=sample.source_ref,
                global_row_index=sample.global_row_index,
                character_count=sample.character_count,
                estimated_tokens=sample.estimated_tokens,
                token_percentile_bucket=sample.token_percentile_bucket.value,
            )
            for sample in report.diagnostic_samples
        ),
    )


def _metrics_to_document(metrics: EmbeddingPerformanceMetrics) -> MetricsJsonDocument:
    return MetricsJsonDocument(
        identity=MetricsIdentityJsonDocument(
            model_id=metrics.model_id,
            model_revision=metrics.model_revision,
            provider=metrics.provider,
            device=metrics.device,
        ),
        dataset=MetricsDatasetJsonDocument(
            record_count=metrics.record_count,
            semantic_text_length_avg=round(metrics.semantic_text_length_avg, 3),
            semantic_text_length_p95=round(metrics.semantic_text_length_p95, 3),
        ),
        tokens=MetricsTokensJsonDocument(
            total_tokens=metrics.total_tokens,
            average_tokens=round(metrics.average_tokens, 3),
            p50_tokens=round(metrics.p50_tokens, 3),
            p95_tokens=round(metrics.p95_tokens, 3),
            p99_tokens=round(metrics.p99_tokens, 3),
            max_tokens=metrics.max_tokens,
        ),
        execution=MetricsExecutionJsonDocument(
            batch_size=metrics.batch_size,
            batches_count=metrics.batches_count,
            embedding_seconds=round(metrics.embedding_seconds, 3),
            records_per_second=round(metrics.records_per_second, 3),
            tokens_per_second=round(metrics.tokens_per_second, 3),
        ),
        hardware=MetricsHardwareJsonDocument(
            gpu_name=metrics.gpu_name,
            cuda_available=metrics.cuda_available,
            peak_memory_mb=(
                round(metrics.peak_memory_mb, 3) if metrics.peak_memory_mb is not None else None
            ),
            average_gpu_utilization_percent=(
                round(metrics.average_gpu_utilization_percent, 3)
                if metrics.average_gpu_utilization_percent is not None
                else None
            ),
        ),
    )


def _batch_latency_to_document(latency: BatchLatencyMeasurement) -> BatchLatencyJsonDocument:
    return BatchLatencyJsonDocument(
        batch_index=latency.batch_index,
        batch_size=latency.batch_size,
        record_count=latency.record_count,
        input_tokens=latency.input_tokens,
        tokens_processed=latency.tokens_processed,
        batch_latency_seconds=round(latency.batch_latency_seconds, 6),
        inference_latency_seconds=round(latency.inference_latency_seconds, 6),
    )

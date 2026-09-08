"""Immutable contracts for embedding performance diagnosis and qualification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

DIAGNOSTIC_MAX_RECORD_LIMIT = 100
TOKEN_P95_REPRESENTATION_THRESHOLD = 2000
TOKEN_P95_REASONABLE_THRESHOLD = 500
LOW_THROUGHPUT_RECORDS_PER_SECOND = 2.0
SIGNIFICANT_BATCH_IMPROVEMENT_RATIO = 1.25
LOW_GPU_UTILIZATION_PERCENT = 50.0
BATCH_EXPERIMENT_SIZES: tuple[int, ...] = (16, 32, 64)


class EmbeddingBottleneckCase(StrEnum):
    REPRESENTATION_OPTIMIZATION = "representation_optimization"
    PROVIDER_OPTIMIZATION = "provider_optimization"
    BATCH_TUNING = "batch_tuning"
    CUDA_PIPELINE = "cuda_pipeline"
    UNDETERMINED = "undetermined"


class TokenPercentileBucket(StrEnum):
    BELOW_P50 = "below_p50"
    P50_TO_P90 = "p50_to_p90"
    P90_TO_P95 = "p90_to_p95"
    P95_TO_P99 = "p95_to_p99"
    AT_OR_ABOVE_P99 = "at_or_above_p99"


class DiagnosticExperimentKind(StrEnum):
    PRODUCTION_BASELINE = "production_baseline"
    BATCH_32 = "batch_32"
    BATCH_64 = "batch_64"
    REPRESENTATION_ONLY = "representation_only"


@dataclass(frozen=True, slots=True)
class CudaEnvironmentSnapshot:
    cuda_available: bool
    gpu_name: str | None


@dataclass(frozen=True, slots=True)
class EmbeddingTokenStatistics:
    count: int
    minimum: int
    mean: float
    p50: float
    p90: float
    p95: float
    p99: float
    maximum: int


@dataclass(frozen=True, slots=True)
class EmbeddingDiagnosticSample:
    record_id: str
    source_ref: str
    global_row_index: int
    character_count: int
    estimated_tokens: int
    token_percentile_bucket: TokenPercentileBucket


@dataclass(frozen=True, slots=True)
class RecordRepresentationMeasurement:
    global_row_index: int
    semantic_text_length: int
    character_count: int
    token_count: int


@dataclass(frozen=True, slots=True)
class BatchLatencyMeasurement:
    batch_index: int
    batch_size: int
    record_count: int
    input_tokens: int
    tokens_processed: int
    batch_latency_seconds: float
    inference_latency_seconds: float


@dataclass(frozen=True, slots=True)
class EmbeddingPerformanceMetrics:
    model_id: str
    model_revision: str
    provider: str
    device: str
    record_count: int
    semantic_text_length_avg: float
    semantic_text_length_p95: float
    total_tokens: int
    average_tokens: float
    p50_tokens: float
    p95_tokens: float
    p99_tokens: float
    max_tokens: int
    batch_size: int
    batches_count: int
    embedding_seconds: float
    records_per_second: float
    tokens_per_second: float
    gpu_name: str | None
    cuda_available: bool
    peak_memory_mb: float | None
    average_gpu_utilization_percent: float | None


@dataclass(frozen=True, slots=True)
class TokenDistributionReport:
    record_count: int
    total_tokens: int
    statistics: EmbeddingTokenStatistics
    semantic_text_length_avg: float
    semantic_text_length_p95: float
    record_measurements: tuple[RecordRepresentationMeasurement, ...]
    diagnostic_samples: tuple[EmbeddingDiagnosticSample, ...]


@dataclass(frozen=True, slots=True)
class EmbeddingExperimentResult:
    experiment_kind: DiagnosticExperimentKind
    batch_size: int
    metrics: EmbeddingPerformanceMetrics
    batch_latencies: tuple[BatchLatencyMeasurement, ...]


BatchExperimentResult = EmbeddingExperimentResult


@dataclass(frozen=True, slots=True)
class EmbeddingDiagnosticClassification:
    case: EmbeddingBottleneckCase
    conclusion: str
    recommended_next_task: str


@dataclass(frozen=True, slots=True)
class EmbeddingDiagnosticReport:
    qualification_id: str
    record_limit: int
    token_distribution: TokenDistributionReport
    baseline: EmbeddingPerformanceMetrics
    batch_experiments: tuple[EmbeddingExperimentResult, ...]
    classification: EmbeddingDiagnosticClassification

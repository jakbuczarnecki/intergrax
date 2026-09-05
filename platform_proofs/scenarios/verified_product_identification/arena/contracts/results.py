"""Arena evidence report contracts."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.arena.contracts.classification import (
    EmbeddingArenaDecision,
    EmbeddingArenaStageStatus,
    EmbeddingArenaVerdict,
    SpeedupBand,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.query_benchmark import (
    EmbeddingArenaQueryCase,
)
from platform_proofs.scenarios.verified_product_identification.qualification.contracts.results import (
    FullBuildEstimate,
    HardwareRuntimeCapability,
    MicrobenchmarkCandidateResult,
    TextLengthStatistics,
    WarmupTimingSnapshot,
)


@dataclass(frozen=True, slots=True)
class ArenaSampleRecordSnapshot:
    offer_id: str
    global_row_index: int
    semantic_text: str
    strata_tags: tuple[str, ...]
    benchmark_only_cluster_id: int | None


@dataclass(frozen=True, slots=True)
class ArenaSampleManifest:
    version: str
    selection_seed: str
    scan_row_limit: int
    target_size: int
    strata_quotas: tuple[tuple[str, int], ...]
    records: tuple[ArenaSampleRecordSnapshot, ...]


@dataclass(frozen=True, slots=True)
class RetrievalQualityMetrics:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    query_count: int


@dataclass(frozen=True, slots=True)
class QualityDeltaMetrics:
    delta_recall_at_1: float
    delta_recall_at_5: float
    delta_recall_at_10: float
    delta_mrr_at_10: float
    delta_ndcg_at_10: float


@dataclass(frozen=True, slots=True)
class TruncationProfile:
    tokenizer_model: str
    max_supported_tokens: int
    truncated_count: int
    truncated_percentage: float
    token_p50: float
    token_p95: float
    token_max: int


@dataclass(frozen=True, slots=True)
class ArtifactSizeEstimate:
    dimension: int
    bytes_per_vector: int
    preliminary_full_artifact_gb: float
    estimation_method: str


@dataclass(frozen=True, slots=True)
class SpeedupEstimate:
    speedup_vs_baseline: float
    speedup_band: SpeedupBand
    hours_saved_vs_baseline: float | None


@dataclass(frozen=True, slots=True)
class QueryLatencySnapshot:
    single_query_p50_seconds: float
    single_query_p95_seconds: float
    small_batch_records_per_second: float | None


@dataclass(frozen=True, slots=True)
class CandidateStageSnapshot:
    stage_name: str
    record_count: int
    status: EmbeddingArenaStageStatus
    selected_provider_batch_size: int | None
    warmup_timing: WarmupTimingSnapshot | None
    microbenchmark_results: tuple[MicrobenchmarkCandidateResult, ...]
    throughput_records_per_second: float | None
    peak_vram_bytes: int | None
    output_dimension: int | None
    detail: str | None


@dataclass(frozen=True, slots=True)
class CandidateRuntimeMetadata:
    provider: str
    model: str
    resolved_revision: str | None
    dimension: int
    input_policy_version: str
    normalization: str
    dtype: str | None
    device: str | None
    batch_size: int | None
    sentence_transformers_version: str | None
    transformers_version: str | None
    torch_version: str | None
    trust_remote_code_required: bool
    requires_remote_code: bool


@dataclass(frozen=True, slots=True)
class CandidateArenaResult:
    candidate_id: str
    verdict: EmbeddingArenaVerdict
    runtime_metadata: CandidateRuntimeMetadata | None
    truncation_profile: TruncationProfile | None
    stage_a: CandidateStageSnapshot | None
    stage_b: CandidateStageSnapshot | None
    stage_c: CandidateStageSnapshot | None
    quality_metrics: RetrievalQualityMetrics | None
    long_input_quality_metrics: RetrievalQualityMetrics | None
    quality_delta_vs_baseline: QualityDeltaMetrics | None
    query_latency: QueryLatencySnapshot | None
    artifact_size_estimate: ArtifactSizeEstimate | None
    full_build_estimate: FullBuildEstimate | None
    speedup_estimate: SpeedupEstimate | None
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EmbeddingArenaReport:
    arena_version: str
    sample_manifest: ArenaSampleManifest
    query_benchmark_version: str
    query_cases: tuple[EmbeddingArenaQueryCase, ...]
    hardware: HardwareRuntimeCapability | None
    text_length_profile: TextLengthStatistics | None
    candidate_results: tuple[CandidateArenaResult, ...]
    decision: EmbeddingArenaDecision
    decision_rationale: str
    finalists_for_5c4c: tuple[str, ...]
    warnings: tuple[str, ...]
    resources_touched: tuple[str, ...]

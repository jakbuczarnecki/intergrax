"""Typed contracts for bounded CUDA batch throughput qualification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding_input_policy import (
    VPI_BGE_M3_DOCUMENT_EFFECTIVE_TOKEN_CEILING,
    VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET,
)

QUALIFICATION_TASK_ID = "VPI-IMPLEMENTATION-5C4E4"
CLOSEOUT_TASK_ID = "VPI-IMPLEMENTATION-5C4E4A"
SAMPLE_RECORD_COUNT = 32
FULL_DATASET_RECORD_COUNT = 3_770_377
MANDATORY_BATCH_VARIANTS: tuple[int, ...] = (1, 4, 8, 16)
OPTIONAL_BATCH_VARIANT = 32
MIN_VRAM_HEADROOM_FRACTION = 0.20
BATCH_32_MIN_IMPROVEMENT_FRACTION = 0.10
CANONICAL_TOKEN_BUDGET = VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET
EFFECTIVE_PROVIDER_TOKEN_CEILING = VPI_BGE_M3_DOCUMENT_EFFECTIVE_TOKEN_CEILING
DATASET_RELATIVE_PATH = (
    "platform_proofs/scenarios/verified_product_identification/dataset/"
    "processed/selected_offers.parquet"
)


class QualificationRunStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    RESOURCE_PRECONDITION_FAIL = "RESOURCE_PRECONDITION_FAIL"


@dataclass(frozen=True, slots=True)
class TokenProfile:
    total_tokens: int
    average_tokens_per_record: float
    p50_tokens: float
    p95_tokens: float
    max_tokens: int


@dataclass(frozen=True, slots=True)
class CudaPreflightSnapshot:
    python_version: str
    torch_version: str
    cuda_runtime_version: str | None
    cuda_available: bool
    gpu_name: str | None
    gpu_total_memory_bytes: int | None
    gpu_free_memory_bytes_before_load: int | None
    gpu_used_memory_bytes_before_load: int | None
    resource_precondition_fail_reason: str | None


@dataclass(frozen=True, slots=True)
class BatchVariantMeasurement:
    batch_size: int
    record_count: int
    total_bounded_input_tokens: int
    average_tokens_per_record: float
    p50_tokens: float
    p95_tokens: float
    max_tokens: int
    wall_clock_embedding_seconds: float
    records_per_second: float
    tokens_per_second: float
    average_milliseconds_per_record: float
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int
    gpu_total_memory_bytes: int
    gpu_free_memory_bytes_before: int
    gpu_free_memory_bytes_after: int
    cuda_oom: bool
    vram_headroom_fraction: float
    safe: bool
    safety_detail: str


@dataclass(frozen=True, slots=True)
class EmbeddingTimeProjection:
    batch_size: int
    records_per_second: float
    projected_seconds: float
    projected_hours: float
    projected_days: float
    safe: bool


@dataclass(frozen=True, slots=True)
class OptionalBatch32Decision:
    executed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ProductionBatchSelection:
    batch_size: int | None
    rationale: str


@dataclass(frozen=True, slots=True)
class BatchVariantQualificationEvidence:
    batch_size: int
    records_per_second: float | None
    safe: bool
    selected: bool
    notes: str


@dataclass(frozen=True, slots=True)
class BoundedCudaThroughputQualificationCloseout:
    """Frozen CUDA throughput qualification decision from VPI-IMPLEMENTATION-5C4E4."""

    closeout_task_id: str
    qualification_task_id: str
    qualified_production_batch_size: int
    qualified_records_per_second: float
    projected_embedding_only_hours: float
    projection_only: bool
    optional_batch_32_executed: bool
    batch_evidence: tuple[BatchVariantQualificationEvidence, ...]


BOUNDED_CUDA_THROUGHPUT_QUALIFICATION_CLOSEOUT = BoundedCudaThroughputQualificationCloseout(
    closeout_task_id=CLOSEOUT_TASK_ID,
    qualification_task_id=QUALIFICATION_TASK_ID,
    qualified_production_batch_size=1,
    qualified_records_per_second=23.42,
    projected_embedding_only_hours=44.7,
    projection_only=True,
    optional_batch_32_executed=False,
    batch_evidence=(
        BatchVariantQualificationEvidence(
            batch_size=1,
            records_per_second=23.42,
            safe=True,
            selected=True,
            notes="SAFE",
        ),
        BatchVariantQualificationEvidence(
            batch_size=4,
            records_per_second=22.36,
            safe=True,
            selected=False,
            notes="NOT SELECTED",
        ),
        BatchVariantQualificationEvidence(
            batch_size=8,
            records_per_second=19.25,
            safe=True,
            selected=False,
            notes="NOT SELECTED",
        ),
        BatchVariantQualificationEvidence(
            batch_size=16,
            records_per_second=19.31,
            safe=True,
            selected=False,
            notes="SAFE but slower than batch 1",
        ),
        BatchVariantQualificationEvidence(
            batch_size=32,
            records_per_second=None,
            safe=False,
            selected=False,
            notes="NOT RUN; gating condition not satisfied",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class BoundedCudaBatchThroughputReport:
    task_id: str
    status: QualificationRunStatus
    python_executable: str
    preflight: CudaPreflightSnapshot
    provider: str
    model: str
    revision: str
    dimension: int
    model_load_count: int
    policy_version: str
    token_budget: int
    dataset_path: str
    record_count: int
    selection_method: str
    token_profile: TokenProfile
    batch_measurements: tuple[BatchVariantMeasurement, ...]
    optional_batch_32: OptionalBatch32Decision
    production_batch_selection: ProductionBatchSelection
    projections: tuple[EmbeddingTimeProjection, ...]
    winner_projection: EmbeddingTimeProjection | None
    projection_only: bool
    oom_observed: bool
    system_instability: bool
    persistent_vram_growth: bool
    known_gaps: tuple[str, ...]

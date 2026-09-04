"""Typed qualification report contracts — scenario-owned evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from platform_proofs.scenarios.verified_product_identification.qualification.contracts.classification import (
    VpiEmbeddingQualificationStatus,
)


QUALIFICATION_VERSION = "5c4a-v1"


class MicrobenchmarkCandidateStatus(str, Enum):
    PASS = "PASS"
    FAILED_OOM = "FAILED_OOM"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class TextLengthStatistics:
    character_min: int
    character_mean: float
    character_p50: float
    character_p95: float
    character_max: int
    token_mean: float | None
    token_p50: float | None
    token_p95: float | None
    token_max: int | None


@dataclass(frozen=True, slots=True)
class HardwareRuntimeCapability:
    python_version: str
    platform: str
    torch_version: str | None
    cuda_available: bool
    cuda_runtime_version: str | None
    gpu_name: str | None
    gpu_count: int
    total_vram_bytes: int | None
    sentence_transformers_version: str | None
    configured_device: str | None
    resolved_provider_device: str | None
    provider_device_proof: str


@dataclass(frozen=True, slots=True)
class EmbeddingIdentitySnapshot:
    provider: str
    model: str
    dimension: int
    embedding_configuration_version: str
    search_representation_derivation_version: str


@dataclass(frozen=True, slots=True)
class ExecutionConfigurationSnapshot:
    device: str | None
    outer_materialization_batch_size: int
    inner_provider_batch_size: int | None
    max_length: int | None
    precision: str


@dataclass(frozen=True, slots=True)
class MicrobenchmarkCandidateResult:
    provider_batch_size: int
    record_count: int
    embed_elapsed_seconds: float
    records_per_second: float
    status: MicrobenchmarkCandidateStatus
    peak_vram_bytes: int | None
    detail: str | None


@dataclass(frozen=True, slots=True)
class WarmupTimingSnapshot:
    provider_init_seconds: float
    first_embed_seconds: float
    steady_embed_seconds: float


@dataclass(frozen=True, slots=True)
class MaterializationQualificationSnapshot:
    state: str
    rows: int
    shards: int
    derive_seconds: float
    embedding_seconds: float
    artifact_write_seconds: float
    total_seconds: float
    embedding_calls: int
    materialization_records_per_second: float
    embedding_records_per_second: float


@dataclass(frozen=True, slots=True)
class RestartQualificationSnapshot:
    state: str
    embedding_calls: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class ArtifactIntegritySnapshot:
    status: str
    detail: str


@dataclass(frozen=True, slots=True)
class StorageQualificationSnapshot:
    attempted: bool
    state: str | None
    catalog_source_rows: int | None
    search_point_count: int | None
    elapsed_seconds: float | None
    detail: str | None


@dataclass(frozen=True, slots=True)
class FullBuildEstimate:
    record_count: int
    estimated_embedding_seconds: float
    estimated_embedding_hours: float
    estimated_derive_seconds: float
    estimated_artifact_write_seconds: float
    estimated_total_seconds: float
    estimated_total_hours: float
    estimation_method: str
    throughput_records_per_second: float
    throughput_source: str


@dataclass(frozen=True, slots=True)
class BottleneckBreakdown:
    derive_share: float
    embedding_share: float
    artifact_write_share: float
    dominant_stage: str
    parallelization_recommendation: str


@dataclass(frozen=True, slots=True)
class VpiEmbeddingQualificationReport:
    qualification_version: str
    status: VpiEmbeddingQualificationStatus
    dataset_path: str
    record_target: int
    microbenchmark_record_count: int
    embedding_identity: EmbeddingIdentitySnapshot
    hardware: HardwareRuntimeCapability
    execution_configuration: ExecutionConfigurationSnapshot
    text_length_profile: TextLengthStatistics
    warmup_timing: WarmupTimingSnapshot
    microbenchmark_results: tuple[MicrobenchmarkCandidateResult, ...]
    selected_provider_batch_size: int | None
    selection_rationale: str
    materialization: MaterializationQualificationSnapshot | None
    materialization_restart: RestartQualificationSnapshot | None
    target_extension_executed: bool
    target_extension_detail: str | None
    artifact_integrity: ArtifactIntegritySnapshot | None
    postgresql_result: StorageQualificationSnapshot
    qdrant_result: StorageQualificationSnapshot
    storage_bootstrap: StorageQualificationSnapshot
    storage_restart: StorageQualificationSnapshot
    zero_storage_embedding_proof: str
    full_build_estimate: FullBuildEstimate | None
    bottleneck: BottleneckBreakdown | None
    warnings: tuple[str, ...]
    resources_touched: tuple[str, ...]

"""Immutable performance profiling contracts for Data Pack pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PipelinePhase(StrEnum):
    """Non-overlapping pipeline phases measured during shard builds."""

    TOTAL = "total"
    READ = "read"
    DERIVE = "derive"
    TOKENIZE = "tokenize"
    EMBEDDING_INFERENCE = "embedding_inference"
    EMBEDDING_BATCH = "embedding_batch"
    PARQUET_WRITE = "parquet_write"
    CHECKSUM = "checksum"
    SOURCE_IDENTITY = "source_identity"
    VALIDATION = "validation"
    STATE_UPDATE = "state_update"


@dataclass(frozen=True, slots=True)
class EmbeddingPerformanceStats:
    model_id: str
    device: str
    batch_size: int
    embedding_calls: int
    embedding_records: int
    embedding_records_per_second: float


@dataclass(frozen=True, slots=True)
class SystemPerformanceStats:
    process_id: int
    started_at: str
    completed_at: str


@dataclass(frozen=True, slots=True)
class ShardPerformanceMetrics:
    shard_ordinal: int
    record_count: int
    read_seconds: float
    derive_seconds: float
    tokenize_seconds: float
    embedding_seconds: float
    write_seconds: float
    validation_seconds: float
    state_update_seconds: float
    total_seconds: float
    checksum_seconds: float
    source_identity_seconds: float
    embedding_batch_seconds: float
    model_id: str
    device: str
    batch_size: int
    embedding_calls: int
    embedding_records: int
    embedding_records_per_second: float
    shard_records_per_second: float
    process_id: int
    started_at: str
    completed_at: str


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    qualification_id: str
    shard_metrics: tuple[ShardPerformanceMetrics, ...]
    dominant_phase: str
    dominant_phase_seconds: float

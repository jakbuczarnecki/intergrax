"""Shard performance metric assembly and bottleneck identification."""

from __future__ import annotations

import os
from datetime import UTC, datetime

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.performance.contracts import (
    EmbeddingPerformanceStats,
    PerformanceReport,
    PipelinePhase,
    ShardPerformanceMetrics,
    SystemPerformanceStats,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.performance.profiler import (
    PipelineProfilerPort,
)

_EMBEDDING_CALLS_COUNTER = "embedding_calls"
_EMBEDDING_RECORDS_COUNTER = "embedding_records"


def record_embedding_batch(
    profiler: PipelineProfilerPort,
    *,
    batch_size: int,
) -> None:
    profiler.increment(_EMBEDDING_CALLS_COUNTER)
    profiler.increment(_EMBEDDING_RECORDS_COUNTER, batch_size)


def build_shard_performance_metrics(
    profiler: PipelineProfilerPort,
    *,
    shard_ordinal: int,
    record_count: int,
    model_id: str,
    device: str,
    batch_size: int,
    started_at: datetime,
    completed_at: datetime,
) -> ShardPerformanceMetrics:
    read_seconds = profiler.seconds(PipelinePhase.READ)
    derive_seconds = profiler.seconds(PipelinePhase.DERIVE)
    tokenize_seconds = profiler.seconds(PipelinePhase.TOKENIZE)
    embedding_inference_seconds = profiler.seconds(PipelinePhase.EMBEDDING_INFERENCE)
    embedding_batch_seconds = profiler.seconds(PipelinePhase.EMBEDDING_BATCH)
    parquet_write_seconds = profiler.seconds(PipelinePhase.PARQUET_WRITE)
    checksum_seconds = profiler.seconds(PipelinePhase.CHECKSUM)
    source_identity_seconds = profiler.seconds(PipelinePhase.SOURCE_IDENTITY)
    validation_seconds = profiler.seconds(PipelinePhase.VALIDATION)
    state_update_seconds = profiler.seconds(PipelinePhase.STATE_UPDATE)
    total_seconds = profiler.seconds(PipelinePhase.TOTAL)

    embedding_calls = profiler.counter(_EMBEDDING_CALLS_COUNTER)
    embedding_records = profiler.counter(_EMBEDDING_RECORDS_COUNTER)
    embedding_seconds = embedding_inference_seconds + tokenize_seconds
    write_seconds = parquet_write_seconds
    embedding_records_per_second = (
        embedding_records / embedding_seconds if embedding_seconds > 0 else 0.0
    )
    shard_records_per_second = record_count / total_seconds if total_seconds > 0 else 0.0

    return ShardPerformanceMetrics(
        shard_ordinal=shard_ordinal,
        record_count=record_count,
        read_seconds=read_seconds,
        derive_seconds=derive_seconds,
        tokenize_seconds=tokenize_seconds,
        embedding_seconds=embedding_seconds,
        write_seconds=write_seconds,
        validation_seconds=validation_seconds,
        state_update_seconds=state_update_seconds,
        total_seconds=total_seconds,
        checksum_seconds=checksum_seconds,
        source_identity_seconds=source_identity_seconds,
        embedding_batch_seconds=embedding_batch_seconds,
        model_id=model_id,
        device=device,
        batch_size=batch_size,
        embedding_calls=embedding_calls,
        embedding_records=embedding_records,
        embedding_records_per_second=embedding_records_per_second,
        shard_records_per_second=shard_records_per_second,
        process_id=os.getpid(),
        started_at=started_at.isoformat(),
        completed_at=completed_at.isoformat(),
    )


def build_performance_report(
    shard_metrics: tuple[ShardPerformanceMetrics, ...],
    *,
    qualification_id: str,
) -> PerformanceReport:
    dominant_phase = "none"
    dominant_seconds = 0.0
    if shard_metrics:
        phase_totals = _aggregate_phase_totals(shard_metrics[-1])
        dominant_phase, dominant_seconds = max(phase_totals.items(), key=lambda item: item[1])
    return PerformanceReport(
        qualification_id=qualification_id,
        shard_metrics=shard_metrics,
        dominant_phase=dominant_phase,
        dominant_phase_seconds=dominant_seconds,
    )


def _aggregate_phase_totals(metrics: ShardPerformanceMetrics) -> dict[str, float]:
    return {
        "read": metrics.read_seconds,
        "derive": metrics.derive_seconds,
        "tokenize": metrics.tokenize_seconds,
        "embedding": metrics.embedding_seconds,
        "embedding_batch": metrics.embedding_batch_seconds,
        "parquet_write": metrics.write_seconds,
        "checksum": metrics.checksum_seconds,
        "source_identity": metrics.source_identity_seconds,
        "validation": metrics.validation_seconds,
        "state_update": metrics.state_update_seconds,
    }


def utc_now() -> datetime:
    return datetime.now(UTC)

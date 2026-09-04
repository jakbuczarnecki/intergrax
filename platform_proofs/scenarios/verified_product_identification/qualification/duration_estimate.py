"""Full-dataset duration estimation from measured steady-state throughput."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.qualification.contracts.results import (
    FullBuildEstimate,
)


def estimate_full_build_duration(
    *,
    record_count: int,
    steady_records_per_second: float,
    derive_seconds_per_record: float,
    artifact_write_seconds_per_record: float,
    throughput_source: str,
) -> FullBuildEstimate:
    if record_count <= 0:
        msg = "record_count must be > 0"
        raise ValueError(msg)
    if steady_records_per_second <= 0.0:
        msg = "steady_records_per_second must be > 0"
        raise ValueError(msg)
    estimated_embedding_seconds = record_count / steady_records_per_second
    estimated_derive_seconds = derive_seconds_per_record * record_count
    estimated_artifact_write_seconds = artifact_write_seconds_per_record * record_count
    estimated_total_seconds = (
        estimated_embedding_seconds
        + estimated_derive_seconds
        + estimated_artifact_write_seconds
    )
    return FullBuildEstimate(
        record_count=record_count,
        estimated_embedding_seconds=estimated_embedding_seconds,
        estimated_embedding_hours=estimated_embedding_seconds / 3600.0,
        estimated_derive_seconds=estimated_derive_seconds,
        estimated_artifact_write_seconds=estimated_artifact_write_seconds,
        estimated_total_seconds=estimated_total_seconds,
        estimated_total_hours=estimated_total_seconds / 3600.0,
        estimation_method=(
            "central estimate from measured steady-state embedding throughput; "
            "derive and artifact write scaled linearly from 1K materialization per-record averages"
        ),
        throughput_records_per_second=steady_records_per_second,
        throughput_source=throughput_source,
    )

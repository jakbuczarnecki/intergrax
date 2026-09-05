"""Artifact size and full-build estimation helpers."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.arena.contracts.results import (
    ArtifactSizeEstimate,
)
from platform_proofs.scenarios.verified_product_identification.qualification.contracts.results import (
    FullBuildEstimate,
)
from platform_proofs.scenarios.verified_product_identification.qualification.duration_estimate import (
    estimate_full_build_duration,
)


def estimate_artifact_size(
    *,
    dimension: int,
    record_count: int,
    bytes_per_float: int = 4,
    overhead_multiplier: float = 1.15,
) -> ArtifactSizeEstimate:
    if dimension <= 0:
        msg = "dimension must be > 0"
        raise ValueError(msg)
    if record_count <= 0:
        msg = "record_count must be > 0"
        raise ValueError(msg)
    bytes_per_vector = dimension * bytes_per_float
    raw_bytes = bytes_per_vector * record_count
    preliminary_gb = (raw_bytes * overhead_multiplier) / (1024.0**3)
    return ArtifactSizeEstimate(
        dimension=dimension,
        bytes_per_vector=bytes_per_vector,
        preliminary_full_artifact_gb=preliminary_gb,
        estimation_method=(
            "PRELIMINARY 1K LINEAR ESTIMATE from dimension * float32 bytes * record_count "
            f"with {overhead_multiplier:.2f}x parquet/metadata overhead"
        ),
    )


def estimate_preliminary_full_build(
    *,
    record_count: int,
    steady_records_per_second: float,
    throughput_source: str,
) -> FullBuildEstimate:
    return estimate_full_build_duration(
        record_count=record_count,
        steady_records_per_second=steady_records_per_second,
        derive_seconds_per_record=0.0,
        artifact_write_seconds_per_record=0.0,
        throughput_source=throughput_source,
    )
